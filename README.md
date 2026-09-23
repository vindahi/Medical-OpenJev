<h1 align="center">Medical-OpenJev 🩺⚡</h1>

<p align="center">
  <img src="web/logo/清华大学-logo-1024px.png" alt="Tsinghua University" height="74">&nbsp;&nbsp;&nbsp;
  <img src="web/logo/山东大学-logo-1024px.png" alt="Shandong University" height="74">&nbsp;&nbsp;&nbsp;
  <img src="web/logo/香港城市大学（东莞）-logo-1024px.png" alt="City University of Hong Kong (Dongguan)" height="74">
</p>

<p align="center">
  <b>Open, locally-deployable, <i>calibrated</i> medical decision gating for three condition tiers — the open answer to closed System-1 APIs like TypeSafe Jev.</b>
</p>

<p align="center">
  <a href="https://huggingface.co/fancc28/Medical-OpenJev"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-HuggingFace-yellow" alt="Models on HuggingFace"></a>
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/models-3%20condition%20tiers-lightgrey" alt="Models">
  <img src="https://img.shields.io/badge/backbone-ModernBERT--base-orange" alt="Backbone">
</p>

---

Modern medical agents put an 8B–70B LLM in front of every trivial decision — *is this evidence enough? which
specialty? should I escalate?* — paying 500–2000 ms and **fake confidence** (`"confidence": 0.95` is just a token
string the model predicted because it *sounds* confident).

**Medical-OpenJev is a System-1 reflex for medical pipelines.** One bidirectional forward pass (tens of ms on a
4090, runs on CPU) returns *routed specialty* + *evidence-sufficiency probability* + *calibrated confidence* +
*act/escalate decision*. It never generates text, so it cannot hallucinate a diagnosis and its confidence is
trained to be **honest** via proper scoring rules (RLCD).

### It's a gate *around* your LLM, not a head *after* it

```
evidence state ──► Medical-OpenJev (System 1, local, ~ms)
                        ├─ act       → trust the fast answer (skip the big LLM)
                        └─ escalate  → only THEN call your general LLM doctor (System 2)
```

Drop-in with **any** general LLM — it never touches the LLM's weights.

---

## 🔥 Headline result

Gate confidence vs. a doctor LLM grading its **own** confidence, on 60 held-out cases per condition tier
(lower ECE = better calibration):

| Condition tier | Gate ECE ↓ | LLM ECE ↓ |
|----------------|:---------:|:---------:|
| Dermatology (`derm`)         | **0.209** | 0.523 |
| Multi-system (`agentclinic`) | **0.135** | 0.351 |
| Rare disease (`rdc`)         | **0.220** | 0.706 |

The gate is **~2–3× better calibrated** than the LLM's gut feeling on every tier.

---

## ✨ Three condition-specific models, one repo

| Model | Condition tier | Example diagnoses |
|-------|----------------|-------------------|
| [`derm/`](derm)        | **Dermatology** — specific, history-grounded | Hailey-Hailey disease, Conradi-Hünermann syndrome |
| [`agentclinic/`](agentclinic) | **Multi-system / acute internal** — broad open-ended workup | Lichen sclerosus, Neuroleptic malignant syndrome |
| [`rdc/`](rdc)          | **Rare disease** (Orphanet) — long-tail, hardest | Hypochondroplasia, Loeffler endocarditis |

A difficulty gradient: well-specified skin disease → broad internal medicine → rare, low-prevalence disease.

## Features

- **3 decision primitives:** `choice` (specialty routing), `noul` (evidence sufficiency), `act/escalate` (gating).
- **Calibrated by design:** RLCD with composite strictly-proper scoring rewards (log / spherical / RPS).
- **Multi-turn friendly:** accumulate consultation evidence turn-by-turn; each turn is one state snapshot.
- **Privacy-first:** fully local, Apache-2.0, no data egress → HIPAA/GDPR-friendly (Jev is a closed cloud API).
- **Lightweight:** frozen ModernBERT-base + tiny trained heads; each tier is one ~300 MB `model.safetensors`.

---

## 🚀 Quickstart

```bash
pip install torch transformers safetensors
huggingface-cli download TODO/medical-openjev --local-dir ./medical-openjev
```

```python
from modeling_medopenjev import MedOpenJev

model = MedOpenJev.from_pretrained("./medical-openjev/derm", device="cuda")  # or agentclinic / rdc / "cpu"

facts = [
    "The patient is a 35-year-old woman.",
    "She is referred for a rash involving both axillae.",
    "She reports recurring episodes since her early 20s.",
    "Lesions develop and involute spontaneously.",
]
state = model.state_from_facts(facts)

print(model.predict(state, domain="derm", qtype="choice"))   # specialty routing
# {'choice': 'Dermatology', 'probs': {...}, 'confidence': ..., 'p_act': ..., 'decision': ...}

g = model.predict(state, domain="derm", qtype="noul")         # evidence-sufficiency gate
print(g["p_true"], g["decision"])                             # P(sufficient), act|escalate
```

### Wrap a general LLM with the gate

```python
g = model.predict(state, domain="rdc", qtype="noul")
if g["decision"] == "act" and g["confidence"] >= 0.85:
    answer = fast_path(state)              # trust System 1
else:
    answer = general_llm_doctor(state)     # escalate to System 2
```

---

## 🧠 How it works

```
state text ("fact1 | fact2 | ...") + typed questions
        │
        ▼
 ModernBERT-base (frozen, bidirectional)          ← one forward pass
        │
   ├── [MASK] per option ─► scorer MLP ─► softmax/T ─► option probabilities
   │                                                └─► confidence = 1 − H(p)/log(K)
   └── [CLS] ⊕ 4 distribution features ─► act head ─► [P(act), P(escalate)]
```

- **RLCD** (REINFORCE w/ group baseline) optimizes a composite **strictly-proper scoring rule** as reward, so the
  model is only maximally rewarded for reporting *honest* probabilities. Backbone frozen; only the heads train.

## 🛠️ Reproduce training & evaluation

```bash
# build per-dataset decision problems (prefix-expanded)
python -m src.data.prefix_expand --dataset derm
# RLCD-train the choice & noul heads (backbone frozen), fit temperature + act threshold on val
python -m src.system1.train_gate --dataset derm --qtype choice --steps 400
python -m src.system1.train_gate --dataset derm --qtype noul   --steps 400
# calibration & gating metrics (E6 / E1)
python -m src.experiments.runner --dataset derm --e e6
# package a single-file HF release
python scripts/merge_release.py --mode per-dataset
```

## 🗂️ Repository layout

```
medical-openjev/
├── modeling_medopenjev.py     # shared loader + predict (inference)
├── derm/         model.safetensors · config.json · medgate_config.json · tokenizer.*
├── agentclinic/  ...
├── rdc/          ...
└── README.md                  # HuggingFace model card
```

> 🚧 **Coming next:** a **unified single model** — all three tiers + both primitives in *one* weight set,
> answering multiple typed questions in a single forward pass (true Jev/Laya parity). It will supersede the three
> per-condition tiers and fix the degenerate `derm/choice` act threshold.

---

## ⚠️ Disclaimer

**Research use only. Not a medical device.** Medical-OpenJev outputs routing and evidence-sufficiency *signals*,
not diagnoses, and must never be used for real clinical decisions or triage of actual patients.

## 📝 Citation

```bibtex
@software{medicalopenjev2026,
  title  = {Medical-OpenJev: Open, Locally-Deployable, Calibrated Medical Decision Gating (three-condition tiers)},
  author = {Ruifan Zuo and Guocheng Hu and Qichao Zhao and Ziyang Meng and Keyv Liu and Zichao Dai and Rui Wang and Tian Gan},
  year   = {2026},
  url    = {https://github.com/vindahi/Medical-OpenJev}
}
```

## Team and affiliations

- **Supervision & Project Lead:** Tian Gan<sup>1</sup>
- **Student Project Leadership:** Ruifan Zuo<sup>1</sup>, Guocheng Hu<sup>1</sup>
- **Specialized Support:**
  - **Medical Support:** Rui Wang<sup>5</sup>
  - **Computing Power Support:** Qichao Zhao<sup>2</sup>
- **Team Members:**
  - Ziyang Meng<sup>3</sup>
  - Keyv Liu<sup>4</sup>
  - Zichao Dai<sup>1</sup>

**Affiliations:** <sup>1</sup> Shandong University · <sup>2</sup> Tsinghua University · <sup>3</sup> City University of Hong Kong · <sup>4</sup> Ocean University of China · <sup>5</sup> Qingdao Endocrine and Diabetes Hospital


## 🙏 Acknowledgements

Conceptually builds on the non-autoregressive System-1 decision paradigm popularized by TypeSafe **Jev** (closed)
and Convai **Laya** (open). Backbone: [`answerdotai/ModernBERT`](https://huggingface.co/answerdotai/ModernBERT-base).

## License

Apache-2.0.
