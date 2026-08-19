import argparse
import json
import os
import time

import bm25s
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "project_config.yaml")
PRODUCTS_PATH = os.path.join(REPO_ROOT, "data", "processed", "products.parquet")
JUDGMENTS_PATH = os.path.join(REPO_ROOT, "data", "processed", "judgments.parquet")
SPLIT_DIR = os.path.join(REPO_ROOT, "data", "processed", "splits")
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_split_query_ids(split):
    path = os.path.join(SPLIT_DIR, f"{split}_query_ids.csv")
    return pd.read_csv(path)["query_id"].tolist()


def build_bm25_index(products):
    corpus_texts = products["product_text"].tolist()

    t0 = time.time()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    tokenize_time = time.time() - t0

    retriever = bm25s.BM25()  # corpus left unset -> retrieve() returns integer positions
    t0 = time.time()
    retriever.index(corpus_tokens, show_progress=False)
    index_time = time.time() - t0

    product_id_array = products["product_id"].to_numpy()
    return retriever, product_id_array, tokenize_time, index_time


def retrieve_top_k(retriever, product_id_array, query_texts, k):
    query_tokens = bm25s.tokenize(query_texts, stopwords="en", show_progress=False)
    result_idx, result_scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
    return product_id_array[result_idx], result_scores


def measure_latency(retriever, query_texts, k):
    latencies = []
    for q in query_texts:
        t0 = time.time()
        q_tok = bm25s.tokenize([q], stopwords="en", show_progress=False)
        retriever.retrieve(q_tok, k=k, show_progress=False)
        latencies.append(time.time() - t0)
    return np.array(latencies) * 1000  # ms


def dcg(relevances):
    relevances = np.asarray(relevances, dtype=float)
    ranks = np.arange(1, len(relevances) + 1)
    return np.sum((2 ** relevances - 1) / np.log2(ranks + 1))


def compute_metrics(query_ids, predicted_ids, judgments):
    query_relevance = {
        qid: dict(zip(g["product_id"], g["relevance"]))
        for qid, g in judgments.groupby("query_id")
    }

    records = []
    for i, qid in enumerate(query_ids):
        preds = predicted_ids[i]
        rel_lookup = query_relevance[qid]

        retrieved_relevant = sum(1 for pid in preds if rel_lookup.get(pid, 0) >= 1)
        total_relevant = sum(1 for r in rel_lookup.values() if r >= 1)
        recall_100 = retrieved_relevant / total_relevant if total_relevant else 0.0

        top10_rel = [rel_lookup.get(pid, 0) for pid in preds[:10]]
        ideal_rel = sorted(rel_lookup.values(), reverse=True)[:10]
        ideal_dcg = dcg(ideal_rel)
        ndcg_10 = dcg(top10_rel) / ideal_dcg if ideal_dcg > 0 else 0.0

        mrr_10 = 0.0
        for rank, pid in enumerate(preds[:10], start=1):
            if rel_lookup.get(pid, 0) >= 1:
                mrr_10 = 1.0 / rank
                break

        records.append({
            "query_id": qid,
            "recall_100": recall_100,
            "ndcg_10": ndcg_10,
            "mrr_10": mrr_10,
        })

    return pd.DataFrame(records)


def run_baseline(split="validation", k=100, products_path=PRODUCTS_PATH, save=True):
    products = pd.read_parquet(products_path)
    judgments = pd.read_parquet(JUDGMENTS_PATH)
    query_ids = load_split_query_ids(split)

    split_judgments = judgments[judgments["query_id"].isin(query_ids)]
    queries_df = (
        split_judgments.drop_duplicates("query_id")[["query_id", "query"]].reset_index(drop=True)
    )

    retriever, product_id_array, tokenize_time, index_time = build_bm25_index(products)

    query_texts = queries_df["query"].tolist()
    predicted_ids, result_scores = retrieve_top_k(retriever, product_id_array, query_texts, k)
    latencies = measure_latency(retriever, query_texts, k)

    metrics_df = compute_metrics(queries_df["query_id"].tolist(), predicted_ids, split_judgments)

    summary = {
        "split": split,
        "products_path": os.path.relpath(products_path, REPO_ROOT),
        "n_queries": len(queries_df),
        "catalogue_size": len(products),
        "recall_100_mean": float(metrics_df["recall_100"].mean()),
        "ndcg_10_mean": float(metrics_df["ndcg_10"].mean()),
        "mrr_10_mean": float(metrics_df["mrr_10"].mean()),
        "recall_100_median": float(metrics_df["recall_100"].median()),
        "ndcg_10_median": float(metrics_df["ndcg_10"].median()),
        "mrr_10_median": float(metrics_df["mrr_10"].median()),
        "latency_ms_median": float(np.median(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "latency_ms_p99": float(np.percentile(latencies, 99)),
        "tokenize_time_s": tokenize_time,
        "index_time_s": index_time,
    }

    if save:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)

        rank_records = [
            {"query_id": qid, "rank": rank, "product_id": pid, "bm25_score": float(score)}
            for i, qid in enumerate(queries_df["query_id"])
            for rank, (pid, score) in enumerate(zip(predicted_ids[i], result_scores[i]), start=1)
        ]
        rankings_df = pd.DataFrame(rank_records)
        rankings_df.to_parquet(os.path.join(ARTIFACTS_DIR, f"bm25_{split}_rankings.parquet"), index=False)

        with open(os.path.join(REPORTS_DIR, "bm25_metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)

    return summary, metrics_df


def main():
    config = load_config()
    default_k = config["data"]["candidate_k"]

    parser = argparse.ArgumentParser(description="Run BM25 retrieval baseline (RankLab Step 3).")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--k", type=int, default=default_k)
    parser.add_argument("--products-path", default=PRODUCTS_PATH)
    args = parser.parse_args()

    summary, _ = run_baseline(split=args.split, k=args.k, products_path=args.products_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
