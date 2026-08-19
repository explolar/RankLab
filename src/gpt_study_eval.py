import json
import os

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from retrieval import ARTIFACTS_DIR, JUDGMENTS_PATH, PRODUCTS_PATH, REPO_ROOT, build_bm25_index, retrieve_top_k

MANUAL_GPT_DIR = os.path.join(REPO_ROOT, "manual_gpt")
REWRITES_PATH = os.path.join(MANUAL_GPT_DIR, "query_rewrites.csv")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")

DENSE_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "indexes", "hnsw_m64_ef256.faiss")
PRODUCT_EMB_IDS_PATH = os.path.join(ARTIFACTS_DIR, "indexes", "product_embedding_ids.csv")
MODEL_NAME = "BAAI/bge-base-en-v1.5"
K = 100


def recall_at_k(predicted_ids, relevant_set):
    if not relevant_set:
        return None
    hit = sum(1 for pid in predicted_ids if pid in relevant_set)
    return hit / len(relevant_set)


def bootstrap_ci(diffs, n_resamples=2000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = np.asarray(diffs)
    means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_resamples)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def main():
    rewrites = pd.read_csv(REWRITES_PATH)
    judgments = pd.read_parquet(JUDGMENTS_PATH)
    products = pd.read_parquet(PRODUCTS_PATH)

    relevant_lookup = {
        qid: set(g.loc[g["relevance"] >= 1, "product_id"])
        for qid, g in judgments.groupby("query_id")
    }

    print("building BM25 index...")
    bm25_retriever, bm25_product_ids, _, _ = build_bm25_index(products)

    raw_texts = rewrites["raw_query"].tolist()
    reviewed_texts = rewrites["reviewed_query"].tolist()

    bm25_raw_ids, _ = retrieve_top_k(bm25_retriever, bm25_product_ids, raw_texts, K)
    bm25_reviewed_ids, _ = retrieve_top_k(bm25_retriever, bm25_product_ids, reviewed_texts, K)

    print("loading dense index + model...")
    dense_index = faiss.read_index(DENSE_INDEX_PATH)
    dense_product_ids = pd.read_csv(PRODUCT_EMB_IDS_PATH)["product_id"].to_numpy()
    model = SentenceTransformer(MODEL_NAME)

    raw_emb = model.encode(raw_texts, batch_size=32, show_progress_bar=False)
    reviewed_emb = model.encode(reviewed_texts, batch_size=32, show_progress_bar=False)
    _, raw_idx = dense_index.search(raw_emb, K)
    _, reviewed_idx = dense_index.search(reviewed_emb, K)
    dense_raw_ids = dense_product_ids[raw_idx]
    dense_reviewed_ids = dense_product_ids[reviewed_idx]

    records = []
    for i, row in rewrites.iterrows():
        qid = row["query_id"]
        relevant = relevant_lookup.get(qid, set())

        bm25_raw_recall = recall_at_k(bm25_raw_ids[i], relevant)
        bm25_reviewed_recall = recall_at_k(bm25_reviewed_ids[i], relevant)
        dense_raw_recall = recall_at_k(dense_raw_ids[i], relevant)
        dense_reviewed_recall = recall_at_k(dense_reviewed_ids[i], relevant)

        records.append({
            "query_id": qid,
            "raw_query": row["raw_query"],
            "reviewed_query": row["reviewed_query"],
            "changed": str(row["raw_query"]).strip().lower() != str(row["reviewed_query"]).strip().lower(),
            "n_relevant": len(relevant),
            "bm25_raw_recall_100": bm25_raw_recall,
            "bm25_reviewed_recall_100": bm25_reviewed_recall,
            "bm25_diff": (bm25_reviewed_recall - bm25_raw_recall) if relevant else None,
            "dense_raw_recall_100": dense_raw_recall,
            "dense_reviewed_recall_100": dense_reviewed_recall,
            "dense_diff": (dense_reviewed_recall - dense_raw_recall) if relevant else None,
        })

    results_df = pd.DataFrame(records)
    results_df.to_parquet(os.path.join(ARTIFACTS_DIR, "gpt_rewrite_study_results.parquet"), index=False)

    valid = results_df.dropna(subset=["bm25_diff", "dense_diff"])
    print(f"queries with >=1 relevant product (evaluable): {len(valid)}/{len(results_df)}")

    summary = {
        "n_queries_total": len(results_df),
        "n_queries_evaluable": len(valid),
        "n_changed": int(results_df["changed"].sum()),
        "n_unchanged": int((~results_df["changed"]).sum()),
    }
    for system in ["bm25", "dense"]:
        diffs = valid[f"{system}_diff"].to_numpy()
        mean_diff, ci_lo, ci_hi = bootstrap_ci(diffs)
        summary[system] = {
            "mean_recall_100_diff": mean_diff,
            "bootstrap_ci_95": [ci_lo, ci_hi],
            "n_positive": int((diffs > 0).sum()),
            "n_negative": int((diffs < 0).sum()),
            "n_zero": int((diffs == 0).sum()),
        }
        for changed_val, label in [(True, "changed_only"), (False, "unchanged_only")]:
            sub = valid[valid["changed"] == changed_val]
            if len(sub) > 0:
                summary[system][f"{label}_mean_diff"] = float(sub[f"{system}_diff"].mean())
                summary[system][f"{label}_n"] = len(sub)

    with open(os.path.join(REPORTS_DIR, "gpt_rewrite_study_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
