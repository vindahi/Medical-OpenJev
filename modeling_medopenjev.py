"""MedOpenJev：单文件发布包的推理加载器。

配合 huggingface/medical-openjev/ 目录使用（model.safetensors + medgate_config.json）：

    from modeling_medopenjev import MedOpenJev
    m = MedOpenJev.from_pretrained("./medical-openjev", device="cuda")
    m.state_from_facts(list_of_str)            # 多轮问诊 -> 当前状态文本
    m.predict(state_text, domain="derm", qtype="noul", options=None)

语义与训练侧一致（src/system1/gate_model.py）：
- 序列模板: "<state> ||| [MASK] opt1 | [MASK] opt2 | ..."，取各 [MASK] 隐状态 -> scorer -> softmax/T；
- 置信度: 1 - H(p)/log(K)（归一化负熵）；
- act 头: [CLS] 拼 4 个分布特征 [top1, top1-top2, 归一化熵, K/255] -> [P(act), P(escalate)]；
- decision: p_act >= act_threshold 则 "act"，否则 "escalate"。
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


def normalized_entropy_conf(probs: List[float]) -> float:
    k = len(probs)
    if k <= 1:
        return 1.0
    h = -sum(p * math.log(p + 1e-12) for p in probs)
    return max(0.0, min(1.0, 1.0 - h / math.log(k)))


def distribution_features(probs: List[float]) -> List[float]:
    ps = sorted(probs, reverse=True)
    k = len(ps)
    top1 = ps[0] if ps else 0.0
    top2 = ps[1] if len(ps) > 1 else 0.0
    ent = -sum(p * math.log(p + 1e-12) for p in ps)
    norm_ent = ent / math.log(k) if k > 1 else 0.0
    return [top1, top1 - top2, norm_ent, k / 255.0]


class MedOpenJev:
    def __init__(self, backbone, tokenizer, heads: Dict[str, Dict], medgate: Dict, device: str):
        self.backbone = backbone.to(device).eval()
        self.tok = tokenizer
        self.heads = heads          # "ds.qtype" -> {"scorer": MLP, "act_head": MLP}
        self.medgate = medgate
        self.device = device
        self.mask_id = tokenizer.mask_token_id

    # ---------- 加载 ----------
    @classmethod
    def from_pretrained(cls, directory: str, device: str = "auto") -> "MedOpenJev":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        mg_path = os.path.join(directory, "medgate_config.json")
        medgate = json.load(open(mg_path))
        from safetensors.torch import load_file
        all_sd = load_file(os.path.join(directory, "model.safetensors"))
        tok = AutoTokenizer.from_pretrained(directory)
        # backbone：从同一 safetensors 的 backbone.* 权重手动加载（存为 bf16）
        cfg = AutoConfig.from_pretrained(directory)
        backbone = AutoModel.from_config(cfg).to(torch.bfloat16)
        bsd = {k[len("backbone."):]: v for k, v in all_sd.items() if k.startswith("backbone.")}
        missing, unexpected = backbone.load_state_dict(bsd, strict=False)
        if unexpected:
            raise RuntimeError(f"backbone 加载存在意外权重: {unexpected[:5]} ...")
        H = medgate["hidden_size"]
        heads = {}
        for key in medgate["heads"]:
            sd = {k.split(f"heads.{key}.", 1)[1]: v for k, v in all_sd.items()
                  if k.startswith(f"heads.{key}.")}
            scorer = nn.Sequential(nn.Linear(H, 256), nn.GELU(), nn.Linear(256, 1))
            act_head = nn.Sequential(nn.Linear(H + 4, 128), nn.GELU(), nn.Linear(128, 2))
            scorer.load_state_dict({k[len("scorer."):]: v.float()
                                    for k, v in sd.items() if k.startswith("scorer.")})
            act_head.load_state_dict({k[len("act_head."):]: v.float()
                                      for k, v in sd.items() if k.startswith("act_head.")})
            scorer = scorer.to(device).eval()
            act_head = act_head.to(device).eval()
            heads[key] = {"scorer": scorer, "act_head": act_head}
        return cls(backbone, tok, heads, medgate, device)

    # ---------- 输入构造 ----------
    @staticmethod
    def state_from_facts(facts: List[str]) -> str:
        """多轮问诊/证据累积 -> 单一状态文本。

        必须与训练侧 src/system1/train_gate.py 的 `" | ".join(state_facts)` 完全一致，
        否则推理输入格式偏离训练分布，校准概率不可信。
        """
        return " | ".join(str(f).strip() for f in facts if str(f).strip())

    # ---------- 推理 ----------
    @torch.no_grad()
    def predict(self, state_text: str, domain: str, qtype: str = "noul",
                options: Optional[List[str]] = None) -> Dict:
        key = f"{domain}.{qtype}"
        if key not in self.heads:
            raise KeyError(f"未知头 {key}，可选: {sorted(self.heads)}")
        if options is None:
            options = (self.medgate["noul_options"] if qtype == "noul"
                       else self.medgate["specialties"])
        head = self.heads[key]
        T = self.medgate["heads"][key]["temperature"]
        thr = self.medgate["heads"][key]["act_threshold"]

        seq = state_text + " ||| " + " | ".join(f"[MASK] {o}" for o in options)
        enc = self.tok(seq, return_tensors="pt", truncation=True, max_length=2048)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hs = self.backbone(**enc).last_hidden_state[0]
        mask_pos = torch.eq(enc["input_ids"][0], self.mask_id).nonzero(as_tuple=True)[0][: len(options)]
        vecs = torch.stack([hs[p] for p in mask_pos]).float()
        logits = head["scorer"](vecs).squeeze(-1) / max(1e-6, T)
        probs = torch.softmax(logits, dim=-1).cpu().tolist()

        feats = distribution_features(probs)
        inp = torch.tensor([hs[0].float().cpu().tolist() + feats], device=self.device)
        ae = torch.softmax(head["act_head"](inp)[0], dim=-1).cpu().tolist()

        idx = max(range(len(probs)), key=lambda i: probs[i])
        conf = normalized_entropy_conf(probs)
        out = {"head": key, "probs": {o: round(p, 4) for o, p in zip(options, probs)},
               "confidence": round(conf, 4),
               "p_act": round(ae[0], 4), "p_escalate": round(ae[1], 4),
               "act_threshold": thr,
               "decision": "act" if ae[0] >= thr else "escalate"}
        if qtype == "choice":
            out["choice"] = options[idx]
        else:
            out["p_true"] = round(probs[-1], 4)   # P(sufficient)
        return out
