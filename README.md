# tiny-skill-linker

This project fine-tunes a 22M-parameter sentence embedder (all-MiniLM-L6-v2) to link short job-ad and résumé sentences to skills. It is evaluated on public, held-out skill-linking benchmarks, and ships as int8 ONNX that runs on CPU in Node or the browser via transformers.js.

The model ranks a skill vocabulary for each sentence by cosine similarity. Because a skill is just a text label, new skills work without retraining.

## Status

| Step | What | Status |
|---|---|---|
| 1 | Calibrate the evaluation harness against the published baseline | done: stock all-mpnet-base-v2 reproduces the paper's RP@5 (39.60 / 26.17 / 33.48 vs 39.6 / 26.2 / 33.5) |
| 2 | Zero-shot baselines, fp32 and int8 | done |
| 3 | Fine-tune MiniLM on synthetic ESCO sentences, 3 seeds | done (fp32) |
| 4 | Ablation: skills held out of training | done (1 seed) |
| 5 | Ablation: weight interpolation (WiSE-FT) vs general retrieval | done: no blend needed |
| 6 | int8 export and regression check | done |

## Results so far

Stock and fine-tuned models, RP@5 and MRR (×100), ranking all 13,891 ESCO 1.1.0 skills per sentence. Test queries: TECH 338, HOUSE 262, TECHWOLF 326.

| Model | Params | Precision | TECH RP@5 | HOUSE RP@5 | TECHWOLF RP@5 | TECH MRR | HOUSE MRR | TECHWOLF MRR |
|---|---|---|---|---|---|---|---|---|
| all-mpnet-base-v2 | 110M | fp32 | 39.60 | 26.17 | 33.48 | 38.76 | 26.27 | 29.58 |
| all-MiniLM-L6-v2 | 22M | fp32 | 45.57 | 33.85 | 36.74 | 43.48 | 33.21 | 35.23 |
| all-MiniLM-L6-v2 | 22M | int8 | 45.56 | 34.31 | 36.17 | 43.29 | 31.90 | 35.06 |
| all-MiniLM-L6-v2 (transformers.js q8 file) | 22M | int8 | 46.78 | 33.69 | 38.24 | 44.20 | 33.17 | 35.61 |
| bge-small-en-v1.5 + query prefix | 33M | fp32 | 44.81 | 31.92 | 34.58 | 44.75 | 32.15 | 33.42 |
| bge-small-en-v1.5 + query prefix | 33M | int8 | 45.30 | 31.04 | 34.76 | 44.55 | 31.16 | 32.91 |
| snowflake-arctic-embed-xs + query prefix | 22M | fp32 | 43.07 | 30.15 | 30.63 | 40.56 | 28.85 | 28.52 |
| snowflake-arctic-embed-xs + query prefix | 22M | int8 | 42.74 | 29.01 | 29.61 | 40.71 | 28.53 | 28.70 |
| **all-MiniLM-L6-v2, fine-tuned (3 seeds, mean ± sd)** | 22M | fp32 | 53.80 ± 0.88 | 46.62 ± 0.93 | 47.66 ± 0.48 | 53.18 ± 0.53 | 45.54 ± 0.78 | 48.02 ± 0.76 |
| **all-MiniLM-L6-v2, fine-tuned (3 seeds, mean ± sd)** | 22M | int8 | 53.79 ± 1.51 | 45.40 ± 0.79 | 46.45 ± 0.84 | 52.81 ± 0.46 | 45.06 ± 1.01 | 47.73 ± 0.80 |
| Decorte et al. best fine-tuned all-mpnet-base-v2 (published) | 110M | fp32 | 54.62 | 45.74 | 54.57 | 52.85 | 42.75 | 52.55 |

- **Calibration:** the paper reports 39.6 / 26.2 / 33.5 RP@5 for stock all-mpnet-base-v2, and this harness reproduces them.
- **Stock MiniLM beats stock mpnet:** at a fifth of the size, it scores higher on all three sets. The baseline for fine-tuning is therefore stock MiniLM, not the paper's stock row.
- **Query prefixes** for bge and arctic were chosen on the TECH and HOUSE validation splits (`results/val/`), by mean RP@5.
- **Fine-tuning** uses `scripts/train.py` with the paper's recipe (MNRL, scale 20, one epoch, batch 64, 2e-5, 5% warmup, sentence-pair augmentation), plus no-duplicate batches. Each seed trains for about 73 minutes on CPU. Against stock MiniLM, every seed improves RP@5 by 7–14 points on every test set; the paired-bootstrap 95% intervals all exclude zero (`scripts/compare.py diff`). The 22M model reaches 98%, 102% and 87% of the paper's 110M fine-tuned RP@5 on TECH, HOUSE and TECHWOLF.
- **Fine-tuned int8:** quantizing costs 0 to 1.2 RP@5 and at most 0.5 MRR (mean over seeds), leaving the int8 model 8 to 12 RP@5 above stock.
- **int8 export:** `scripts/export_int8.py` uses per-channel weights with reduced range. That setting's embeddings match the transformers.js q8 file most closely (mean cosine 0.988 over 2,000 ESCO skill names, not test data), and it stays within 0.6 RP@5 of fp32.

### Skills never seen in training

`ft-holdout-seed0` is trained with the same recipe, but 2,765 of the 13,826 synthetic-data skills (20%, chosen at random) and all of their sentences are removed. On the test sets, each (sentence, gold skill) pair is then scored as seen or unseen by that model. Hit@5 means the gold skill is ranked in the top 5 of 13,891. The results are from `scripts/unseen.py`.

| Model | Seen skills hit@5 (n=1,414) | Unseen skills hit@5 (n=282) |
|---|---|---|
| all-MiniLM-L6-v2, stock | 31.40 | 33.69 |
| Fine-tuned on all skills (seed 0) | 41.73 | 43.26 |
| Fine-tuned with those skills held out (seed 0) | 41.44 | 41.84 |

- **Against stock,** the held-out model gains +8.16 hit@5 on skills it never saw (95% CI [+3.55, +12.77], paired bootstrap over pairs), against +10.04 on skills it saw.
- **Against the model that saw those skills,** it is 1.42 hit@5 lower on them (95% CI [−3.55, +0.71]). Most of the fine-tuning gain therefore carries over to skills with no training sentences, which is what an open tag vocabulary needs.
- **Caveat:** this is a single seed, and only 282 test pairs involve held-out skills.

### Weight interpolation and general retrieval

Seed 1 (best mean validation RP@5) blended with stock MiniLM, θ(α) = (1−α)·θ_stock + α·θ_fine-tuned. The rule, fixed before the sweep, picks the best mean validation RP@5 among blends that keep at least 95% of stock's NanoBEIR NDCG@10.

| α | Validation RP@5, TECH | HOUSE | Mean | NanoBEIR NDCG@10 | Share of stock's |
|---|---|---|---|---|---|
| 0 (stock) | 53.04 | 48.14 | 50.59 | 56.24 | 100.0% |
| 0.25 | 62.38 | 51.97 | 57.17 | 56.30 | 100.1% |
| 0.5 | 66.93 | 56.07 | 61.50 | 56.10 | 99.8% |
| 0.75 | 67.71 | 55.49 | 61.60 | 55.81 | 99.2% |
| 1 (seed 1) | 69.71 | 57.54 | 63.63 | 55.11 | 98.0% |

Fine-tuning costs 2% of stock's NanoBEIR score, and every blend trades validation RP@5 for less than that, so α = 1 is chosen and needs no further test scoring.

### Transfer to a private résumé-tagging set

This check scores a different task: 68 private résumé bullets, each to be tagged from 26 broad, hyphenated category tags (such as `system-architecture` and `leadership`). It has a single annotator. The text is not published; the set's SHA-256 begins `5db062ebb3c52b39`. Both models are int8 in transformers.js, ranking bare tag names. The check was run once, after the model was chosen.

| Model | hit@1 | hit@3 | MRR |
|---|---|---|---|
| all-MiniLM-L6-v2, stock (transformers.js q8) | 0.603 | 0.868 | 0.735 |
| Fine-tuned seed 1, int8 | 0.559 | 0.809 | 0.705 |

The fine-tuned model is not better on this set. Against stock, hit@3 is −0.059 (95% CI [−0.133, 0.000], cluster bootstrap) and MRR −0.031 ([−0.084, +0.021]). By bullet, it is better on 1 and worse on 5 (McNemar p = 0.22). It gains on tags named like concrete skills (`latex-documents`, `testing`, `model-evaluation`) and loses on broad categories (`system-architecture`, `backend-api`, `documentation`). That fits training on fine-grained ESCO skill names. Skill linking improves, but the gain doesn't carry over to coarse category tagging, so the deployed tagger keeps the stock model.

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
