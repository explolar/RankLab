# RankLab

## Problem

E-commerce search ranking is a two-stage problem — retrieve a candidate set, then rank it — and each stage has a different failure mode: retrieval determines the *ceiling* (a reranker can't recover a product that was never retrieved), ranking determines how well that ceiling gets realized in the top of the page. RankLab builds and measures both stages on real data (Amazon's ESCI Shopping Queries dataset), then asks two further questions on top: does automated LLM assistance (query rewriting, relevance judging) measurably help, and does a synthetic A/B experiment framework correctly avoid the standard statistical traps (peeking early, uncorrected multiple comparisons, unexplained variance)?

**RankLab's core search pipeline (retrieval → features → ranking) is fully offline and local — no API, no hosted LLM, no backend service in that path.** The deliberate exceptions are Steps 9 and 10's LLM studies (query rewriting and relevance judging), which use the Groq API for generation (a documented deviation from this project's original manual-only design — see "GPT/automated LLM studies" below and `reports/results_log.md`'s 2026-08-19 methodology-change entries for the full reasoning). Results from those studies are labeled by their actual provenance, not presented as either "manual GPT" or as a scalable online system. Steps 11-12's experiment simulator is entirely local and, separately, entirely synthetic — every result there is explicitly labeled as simulated, not a claim about real customer behavior.

Dated experiment log, the authoritative source for every number in this README: [`reports/results_log.md`](reports/results_log.md).

**Status:** Steps 1-12 of a 13-step build plan are done (data prep, retrieval, ranking, both LLM studies, and the synthetic experiment simulator); Step 13 (packaging/demo) is this README, `app.py`, and the run_all guide below.

## Data

Amazon [ESCI Shopping Queries dataset](https://github.com/amazon-science/esci-data), US locale, `small_version=1` slice. 601,354 judgment rows, 29,844 queries, split 70/15/15 by `query_id` (seed=42, stratified by query-level mean relevance) so no query leaks across train/validation/test.

Catalogue: full 482,105-product US catalogue. A smaller 164,006-product reduction was used temporarily for Steps 4-5 (local GPU, GTX 1650 4.3GB, made full-catalogue dense embedding impractical to iterate on at first) but was unified back to the full catalogue before Step 6 — a 164k-only catalogue left 38.6% of train queries with zero relevant-product coverage, unusable for LambdaMART training. Full story: `reports/results_log.md`, "Catalogue unified back to full 482k" entry.

The raw dataset (`esci-data/`) and all derived data/pipeline artifacts (`data/`, `artifacts/`) are **not** committed to this repo — see `.gitignore` and the run_all guide below. They're multi-gigabyte and fully regeneratable; only the code, configs, small measured-result summaries (`reports/`), and the LLM-study data (`manual_gpt/`) are tracked.

## System diagram

```text
Amazon ESCI queries + products + human relevance labels
                         |
                         v
      Local data preparation and train/validation/test split by query
                         |
        +----------------+----------------+
        |                                 |
        v                                 v
  BM25 lexical retrieval           bge-base-en-v1.5 embeddings
   (src/retrieval.py)              + HNSW index (src/candidates.py)
        |                                 |
        +------------ candidate pool (BM25 top-100 UNION dense top-100) ------------+
                                                           |
                                                           v
                    14-feature extraction (src/feature_extraction.py)
                          + LightGBM LambdaMART (src/ranker.py)
                                                           |
                                                           v
                         ranked top 10 + offline metrics
                                                           |
              +--------------------+---------------------+
              |                                          |
              v                                          v
  GPT/automated LLM study                       Synthetic A/B/n simulator
  (rewrites / judging, Groq API,                (src/simulation.py --
   human-reviewed -- Steps 9-10)                 CUPED + sequential tests, Steps 11-12)
```

## Reproduce (run_all guide)

No API key is required for the core pipeline (retrieval → candidates → features → ranking → simulator). `src/automated_rewrite.py` (Step 9) and `src/llm_judging.py` (Step 10) need `GROQ_API_KEY` set in your own environment if you want to *regenerate* the LLM-study data from scratch — get a free key at console.groq.com; never commit it or paste it into a chat. The LLM-study outputs already exist under `manual_gpt/` and don't need regenerating just to explore the results.

**1. Clone and install:**
```bash
git clone <this-repo-url>
cd RankLab
pip install -r requirements.txt
git clone https://github.com/amazon-science/esci-data
```
Then download the ESCI parquet files into `esci-data/shopping_queries_dataset/` per that repo's own instructions (they're too large to be part of either repo's git history).

**2. One-time data prep (notebooks, in order — builds `data/processed/` and `artifacts/indexes/` from the raw dataset):**
`01_eda_data_inspection.ipynb` → `02_retrieval_baseline.ipynb` → `03_dense_embeddings.ipynb` → `04_retrieval_index_benchmark.ipynb`

Note: Step 3 (dense embeddings) is the slow one — expect anywhere from ~10 minutes to a few hours depending on your GPU, since it's encoding hundreds of thousands of products. See `reports/results_log.md` if you hit memory constraints on a smaller GPU; the notebook documents the exact workaround used here (encoding in a reduced-catalogue pass first, then filling in the rest).

**3. Core pipeline (scripts, in order):**
```bash
python src/retrieval.py --split validation           # BM25 baseline
python src/candidates.py --split all                 # BM25 union dense candidate pools, all splits
python src/feature_extraction.py --split all         # 14 features per candidate row
python src/ranker.py --grid-search --evaluate-test   # train + tune LambdaMART, evaluate once on test
```

**4. LLM studies (optional — needs `GROQ_API_KEY`; pre-generated results are already tracked in git under `manual_gpt/`, so this step is only needed to regenerate them from scratch):**
```bash
python src/automated_rewrite.py       # Step 9: query rewrite generation
python src/llm_judging.py             # Step 10: relevance judging generation
python src/gpt_study_eval.py          # Step 9 evaluation: paired retrieval + bootstrap CI
```

**5. Synthetic experiment (Steps 11-12 — needs the ranker and the Step 9 rewrites from steps 3-4 above):**
```bash
python src/build_experiment_arms.py       # rebuild the 3 ranked-list arms (~2-3 min, no retraining)
python src/simulation.py --stability-seeds 100
```

**6. Interactive demo:**
```bash
python -m streamlit run app.py
```
(use `python -m streamlit`, not bare `streamlit` — pip installs the CLI script outside PATH on some setups, this form always works)
Pick an example query or type your own; shows BM25/dense/LambdaMART top-10 side by side, the LambdaMART feature breakdown, and — if the query happens to be one of the 150 Step-9 rewrite-study queries — the actual reviewed rewrite, clearly labeled with its real provenance (automated generation, human-reviewed).

## Measured results

### Retrieval

- **BM25** (`bm25s`, over title+brand+bullets+color), full 482k catalogue: Recall@100 0.460, NDCG@10 0.311, MRR@10 0.553 (`src/retrieval.py`).
- **Dense** (`BAAI/bge-base-en-v1.5`, 768-dim), full 482k catalogue: FlatIP (exact) Recall@100 0.509, p95 latency 75.3ms. **Chosen index: HNSW M=64, efSearch=256** — Recall@100 0.500 (98.2% of exact), p95 latency **4.5ms (~16.7x faster than exact search at this scale)**, 1.7GB index.

**Why HNSW over FlatIP:** every benchmarked configuration cleared any reasonable latency/memory budget trivially at this catalogue size — the recall/latency/memory trade-off the exercise is meant to surface doesn't actually bind at 482k vectors. HNSW was chosen anyway, over FlatIP, to demonstrate the approximate-nearest-neighbor operating-point methodology that becomes load-bearing at production scale (millions of vectors), not because FlatIP was too slow here. IVF-PQ skipped: no memory pressure to solve. Config-selection grid (`reports/retrieval_benchmark.csv`, chart: `reports/figures/retrieval_pareto.png`) was run at the intermediate 164k catalogue as an initial screen (Section 3e); the *chosen* configuration was then re-validated at the full 482k scale (Section 3f) rather than re-running the entire 12-config grid a second time at full scale.

Dense retrieval beats BM25 on every metric — expected from BM25's own failure analysis (`reports/results_log.md`, Step 3 entry), which found BM25's dominant failure mode is negation ("trash can *without* lid" retrieves cans *with* lids) and lexical/synonym gaps dense embeddings are structurally built to catch.

### Ranking

LightGBM `LGBMRanker` (`lambdarank` objective) on 14 hand-built features (`src/feature_extraction.py`, `reports/feature_dictionary.json`) over a BM25∪dense candidate pool (`src/candidates.py`). Small tuning grid (`src/ranker.py --grid-search`), evaluated on held-out test (touched once):

| System | Test Recall@10 | Test NDCG@10 | Test MRR@10 |
|---|---:|---:|---:|
| BM25 (candidate-pool-scoped) | 0.193 | 0.308 | 0.553 |
| Dense | 0.196 | 0.317 | 0.549 |
| Naive candidate-union (normalized score sum) | 0.210 | 0.337 | 0.590 |
| **LambdaMART** | **0.224** | **0.365** | **0.613** |

Chart: `reports/figures/ranking_comparison.png`.

LambdaMART beats BM25 by +18.5% relative NDCG@10, and — the more meaningful comparison — beats the naive union baseline by +8.3%, showing the *learned* feature combination adds value beyond just summing the two retrieval signals. One custom feature, `negation_conflict_flag`, directly operationalizes the BM25 negation failure found in Step 3 and shows real negative correlation with relevance (-0.10) and real usage in the trained model (195 tree splits) — not just a theoretical addition. Full writeup: `reports/results_log.md`, Step 8 entry.

## GPT/automated LLM studies (Steps 9-10)

Both studies use the Groq API (`openai/gpt-oss-120b`) for generation rather than a manual GPT chat interface — the user's explicit choice for speed, shown the trade-off each time (`reports/results_log.md` has the full reasoning for both). The model was originally meant to be Llama; Groq no longer hosts a general-purpose Llama chat model (confirmed live against Groq's `/models` endpoint), so it's `gpt-oss-120b` throughout, and every row's `model_label` records the actual model string used — never silently mislabeled.

**Step 9 (query rewrite) protocol:** 150 held-out short queries (test-split, 1-2 tokens) sampled and frozen (`manual_gpt/rewrite_study_query_ids.csv`) *before* any generation. Fixed prompt: `manual_gpt/prompts/rewrite_v1.md`. Raw output preserved unedited in `manual_gpt/query_rewrites.csv`. Human review pass (accept/edit/reject) preserved — 120/150 were unchanged rewrites (nothing to review, since nothing was altered), 30/150 genuinely reviewed and accepted. Evaluated with paired local retrieval (raw vs. reviewed query) + bootstrap CI (2,000 resamples): **BM25 Recall@100 lift +0.013 (95% CI [0.003, 0.025] — excludes 0, real)**, dense lift +0.015 (CI [0.0004, 0.034] — barely excludes 0, marginal). Rewriting helps lexical retrieval more than semantic retrieval, which makes sense mechanistically (BM25 needs literal token overlap; dense embeddings already capture meaning from the raw query).

**Step 10 (relevance judging) protocol:** 220 blinded query-product pairs (test-split, stopped early from a planned 400, still within the plan's 200-400 range), balanced across ESCI's four relevance grades, labels hidden from the prompt throughout (`manual_gpt/judging_study_blinded.csv` vs. `judging_study_hidden_labels.csv`, joined only after generation). Fixed prompt: `manual_gpt/prompts/judging_v1.md`. **Weighted (linear) Cohen's kappa = 0.478** (moderate agreement, Landis-Koch scale). Two systematic disagreement patterns found with concrete examples, not just a bare confusion matrix: the model is stricter about brand/attribute matching than some ESCI "Exact" labels (e.g. `"supreme brand clothing"` matched to a Champion-brand tee, correctly flagged as a brand mismatch), and applies a stricter literal-accessory standard than ESCI's looser "Complement" convention (e.g. `"bong"` matched to a cookbook titled *"Bong Appétit"* — wordplay, not a real accessory).

Real infrastructure lessons from building these (both fixed, both now in shared `src/llm_client.py`): a data-loss bug (results weren't saved until a run's very end — now incremental, resumable), and a rate-limit backoff that reacted to failures instead of watching Groq's actual token budget (now proactive).

## Experiment simulation (Steps 11-12)

Three arms — A=BM25, B=LambdaMART, C=LambdaMART+rewrite-policy — compared via a simulated click experiment. **Arm C required a genuine rerun** (fresh retrieval + features + LambdaMART scoring on the 150 reviewed rewritten queries, using already-built indices/model, no retraining), not a shortcut. Every simulation parameter (attraction probabilities, click rule, user-correlation mechanism) is an explicit, documented assumption in `configs/project_config.yaml`, not observed behavior — every chart says "SYNTHETIC CLICK SIMULATION" in its title.

Sanity checkpoint passed (CTR rises with relevance strength: A < B ≈ C). Both B and C beat A with real effect sizes (+4.6pp / +6.3pp CTR, 95% CIs exclude zero, BH-significant in **100/100** stability seeds — not a fluke). B-vs-C's CI includes zero and is correctly *not* claimed as a real difference. Sequential O'Brien-Fleming testing correctly declines to stop early even when a naive unadjusted p-value dips below 0.05 partway through (25% info fraction), and CUPED reduces variance ~15-16%. Full writeup, decision table, and charts: `reports/results_log.md` (Steps 11+12 entry), `reports/experiment_decision_table.csv`, `reports/figures/sequential_analysis.png`, `reports/figures/cuped_comparison.png`.

## Limitations

- Steps 9-10's LLM outputs are gpt-oss-120b-generated via the Groq API, not manually collected via a GPT chat interface as the project originally specified. Human review/blinding is still preserved in both.
- LightGBM training uses only judged candidate-pool rows (238,623 train rows) per the plan's explicit policy — unjudged candidates are never treated as implicit negatives, so the model doesn't see the full candidate pool's scale during training, only at inference.
- The synthetic experiment's click model (attraction probabilities, examination formula, user-correlation mechanism) is an explicit documented assumption, not fit to or validated against any real click data — the point is demonstrating the statistical methodology (CUPED, sequential testing, multiple-comparison correction), not predicting real CTR.
- The retrieval index-selection *grid* (12 HNSW configs) was run once, at the intermediate 164k catalogue, not repeated at full 482k scale — only the chosen configuration was re-validated at full scale. The grid's qualitative conclusion (approximation cost is negligible at this catalogue size) is very unlikely to change at 482k vs 164k vectors, but the full 12-point curve at exact full-scale numbers doesn't exist.
- The optional Step 10 judging study was stopped at 220/400 pairs (mid-run, for time) — still within the plan's stated 200-400 range, with reasonable class balance preserved, but with less statistical power than the full 400 would have given.

## Next steps

- Step 10's manual QA checklist item on kappa confidence intervals (only the point estimate is currently reported) could be added via bootstrap, matching the rigor already applied to Step 9's CI.
- The Step 11-12 experiment simulator's click model is a placeholder assumption; if real click-through data ever becomes available for this catalogue, the attraction-probability-by-grade config could be calibrated against it instead of asserted.
- Phase D's arms are currently BM25 / LambdaMART / LambdaMART+rewrite; a fourth arm folding in Step 10's judging-agreement findings (e.g. a rescoring adjustment for the systematic Complement-mislabeling pattern) would be a natural extension, not yet built.
- The full 12-config HNSW grid could be re-run at 482k scale for a completely current Pareto chart, if the extra compute time (last estimated at over an hour for the full latency-loop sweep) is worth it for a project at this stage — currently judged not worth it given the chosen config was already re-validated.
