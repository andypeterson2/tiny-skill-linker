# Plan

## Question

Can a 22M-parameter embedder, fine-tuned on public synthetic data and quantized to int8 for CPU inference, link sentences to skills nearly as well as a published 110M fine-tuned model? And does it keep working for skills it never saw in training?

## Reference numbers

Decorte et al. (2023) report RP@5 when ranking all ESCO skills:

| Model | TECH | HOUSE | TECHWOLF |
|---|---|---|---|
| all-mpnet-base-v2 (110M), stock | 39.6 | 26.2 | 33.5 |
| Their best fine-tuned model | 54.6 | 45.7 | 54.6 |

These numbers are cited here only if step 1 reproduces the stock row within about one point in this harness.

## Method

- **Training:** full fine-tuning with sentence-transformers' `SentenceTransformerTrainer`. The loss is `MultipleNegativesRankingLoss`, with `GISTEmbedLoss` as a comparison, and batches use `BatchSamplers.NO_DUPLICATES`.
- **Model selection:** the TECH and HOUSE validation splits only.
- **Hardening:** hard negatives, and WiSE-FT interpolation between the base and fine-tuned weights to keep general retrieval ability.
- **Export:** `export_dynamic_quantized_onnx_model` to int8, rescored after quantization.

## Reporting

- **Per test set:** point estimate and mean ± sd over 3 seeds (never the best seed).
- **Against the stock model:** a paired-bootstrap 95% CI on the difference.
- **Both precisions:** fp32 and int8.
- **General retrieval:** NanoBEIR retention.
- **Private check:** a private in-domain résumé set is used only as a one-off transfer check. Its text is never committed, and it is never used for selection.

## Steps
1. **Freeze the harness.** Fix the WorkRB metric, ESCO version, candidate set and filtering. Calibrate by reproducing the stock all-mpnet-base-v2 row (39.6 / 26.2 / 33.5) within about a point. Commit the scripts, pinned versions and config, and record dataset revisions.
2. **Baselines, fp32 and int8:** MiniLM-L6, arctic-embed-xs (with its query prefix) and bge-small (33M). Table with parameter counts and int8 latency.
3. **Headline run:** MiniLM-L6 on Synthetic-ESCO with MNRL. Choose epochs and learning rate on TECH + HOUSE validation, run 3 seeds, and score the three tests once. Report vs stock with a paired-bootstrap CI (bootstrap by job ad if possible), and as "Z% of the paper's 110M fine-tuned row".
4. **Ablation A:** hold out 20% of ESCO skills with the same recipe, and report RP@5 on seen vs unseen gold skills.
5. **Ablation B:** sweep the WiSE-FT α on validation against NanoBEIR retention. If budget remains, one GIST vs MNRL run.
6. **Export and transfer check:** int8 export, rescore, NanoBEIR regression. Run stock and the chosen checkpoint once on the private in-domain set, after a second annotator pass with agreement reported. Ship only if hit@1 and MRR don't regress beyond the CI.
7. **Write-up and model card** with attribution (TechWolf CC-BY-4.0, Decorte et al., ESCO version, MiniLM Apache-2.0). No private evaluation text in this repo, and no training after this point.

**Budget:** at most 10 runs. Early-stop when validation RP@5 is flat over two evals. Once any test set is scored, no more sweeps.
