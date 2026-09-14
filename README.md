# tiny-skill-linker

This project fine-tunes a 22M-parameter sentence embedder (all-MiniLM-L6-v2) to link short job-ad and résumé sentences to skills. It is evaluated on public, held-out skill-linking benchmarks, and ships as int8 ONNX that runs on CPU in Node or the browser via transformers.js.

The model ranks a skill vocabulary for each sentence by cosine similarity. Because a skill is just a text label, new skills work without retraining.

## Status

The plan is in [docs/plan.md](docs/plan.md). No results yet.

| Step | What | Status |
|---|---|---|
| 1 | Calibrate the evaluation harness against the published baseline | done: stock all-mpnet-base-v2 reproduces the paper's RP@5 (39.60 / 26.17 / 33.48 vs 39.6 / 26.2 / 33.5) |
| 2 | Zero-shot baselines, fp32 and int8 | not started |
| 3 | Fine-tune MiniLM on synthetic ESCO sentences, 3 seeds | not started |
| 4 | Ablation: skills held out of training | not started |
| 5 | Ablation: weight interpolation (WiSE-FT) vs general retrieval | not started |
| 6 | int8 export and regression check | not started |

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps workrb==0.6.0
.venv/bin/pip install --no-deps -e .
```

`requirements.lock` pins every version the results were produced with. torch 2.2.2 is the last release built for Intel Macs. WorkRB declares torch ≥ 2.6, but its evaluation code runs on 2.2.2; the calibration in step 1 checks that.

## Data and attribution

- **Evaluation:** TechWolf's ESCO skill-linking test sets (TECH, HOUSE, TECHWOLF), CC-BY-4.0, run through [WorkRB](https://github.com/techwolf-ai/WorkRB).
- **Training:** TechWolf's Synthetic-ESCO-skill-sentences, CC-BY-4.0, from Decorte et al., *Extreme Multi-Label Skill Extraction Training using Large Language Models* (2023), [arXiv:2307.10778](https://arxiv.org/abs/2307.10778).
- **Skill taxonomy:** [ESCO](https://esco.ec.europa.eu/) (European Skills, Competences, Qualifications and Occupations), © European Union. The version used is recorded with each result.
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0.

Datasets and model weights are downloaded at run time and never committed.

## License

MIT (code). Datasets and models keep their own licenses, listed above.
