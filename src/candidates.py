import argparse
import json
import os
import time

import faiss
import numpy as np
import pandas as pd

from retrieval import (
    ARTIFACTS_DIR,
    JUDGMENTS_PATH,
    PRODUCTS_PATH,
    REPO_ROOT,
    build_bm25_index,
    load_split_query_ids,
    retrieve_top_k as bm25_retrieve_top_k,
)

INDEXES_DIR = os.path.join(ARTIFACTS_DIR, "indexes")
DENSE_INDEX_PATH = os.path.join(INDEXES_DIR, "hnsw_m64_ef256.faiss")
PRODUCT_EMB_IDS_PATH = os.path.join(INDEXES_DIR, "product_embedding_ids.csv")
QUERY_EMB_PATH = os.path.join(INDEXES_DIR, "query_embeddings.npy")
QUERY_EMB_IDS_PATH = os.path.join(INDEXES_DIR, "query_embedding_ids.csv")


def load_dense_index():
    index = faiss.read_index(DENSE_INDEX_PATH)
    product_id_array = pd.read_csv(PRODUCT_EMB_IDS_PATH)["product_id"].to_numpy()
    return index, product_id_array


def load_query_embeddings():
    embeddings = np.load(QUERY_EMB_PATH)
    query_ids = pd.read_csv(QUERY_EMB_IDS_PATH)["query_id"].to_numpy()
    return embeddings, query_ids


def dense_retrieve(index, product_id_array, query_embeddings, k, chunk_size=500):
    all_scores, all_idx = [], []
    for start in range(0, len(query_embeddings), chunk_size):
        chunk = query_embeddings[start:start + chunk_size]
        scores, idx = index.search(chunk, k)
        all_scores.append(scores)
        all_idx.append(idx)
    scores = np.vstack(all_scores)
    idx = np.vstack(all_idx)
    return product_id_array[idx], scores


def _to_long(predicted_ids, scores, query_ids, score_col, source_col):
    k = predicted_ids.shape[1]
    return pd.DataFrame({
        "query_id": np.repeat(query_ids, k),
        "product_id": predicted_ids.flatten(),
        score_col: scores.flatten(),
        source_col: True,
    })


def build_candidates_for_split(split, judgments, bm25_retriever, bm25_product_ids,
                                dense_index, dense_product_ids, query_embeddings, query_emb_ids, k=100):
    query_ids = load_split_query_ids(split)
    split_judgments = judgments[judgments["query_id"].isin(query_ids)]
    queries_df = split_judgments.drop_duplicates("query_id")[["query_id", "query"]].reset_index(drop=True)

    bm25_predicted_ids, bm25_scores = bm25_retrieve_top_k(
        bm25_retriever, bm25_product_ids, queries_df["query"].tolist(), k
    )

    emb_lookup = {qid: i for i, qid in enumerate(query_emb_ids)}
    emb_rows = [emb_lookup[qid] for qid in queries_df["query_id"]]
    split_query_embeddings = query_embeddings[emb_rows]
    dense_predicted_ids, dense_scores = dense_retrieve(dense_index, dense_product_ids, split_query_embeddings, k)

    qid_array = queries_df["query_id"].to_numpy()
    bm25_long = _to_long(bm25_predicted_ids, bm25_scores, qid_array, "bm25_score", "from_bm25")
    dense_long = _to_long(dense_predicted_ids, dense_scores, qid_array, "dense_score", "from_dense")

    candidates = pd.merge(bm25_long, dense_long, on=["query_id", "product_id"], how="outer")
    candidates["from_bm25"] = candidates["from_bm25"].fillna(False)
    candidates["from_dense"] = candidates["from_dense"].fillna(False)

    candidates = candidates.merge(
        split_judgments[["query_id", "product_id", "relevance"]],
        on=["query_id", "product_id"], how="left",
    )
    candidates["is_judged"] = candidates["relevance"].notna()

    candidates = candidates.sort_values("query_id").reset_index(drop=True)
    group_sizes = candidates.groupby("query_id").size()
    assert group_sizes.sum() == len(candidates), "group sizes don't sum to candidate row count"

    return candidates, queries_df, group_sizes


def summarize(split, candidates, queries_df, group_sizes):
    both = ((candidates["from_bm25"]) & (candidates["from_dense"])).sum()
    return {
        "split": split,
        "n_queries": len(queries_df),
        "n_candidate_rows": len(candidates),
        "mean_candidates_per_query": float(group_sizes.mean()),
        "n_judged": int(candidates["is_judged"].sum()),
        "pct_judged": float(candidates["is_judged"].mean()),
        "pct_from_both": float(both / len(candidates)),
        "pct_from_bm25_only": float(((candidates["from_bm25"]) & (~candidates["from_dense"])).mean()),
        "pct_from_dense_only": float(((~candidates["from_bm25"]) & (candidates["from_dense"])).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Build BM25+dense union candidate pools (RankLab Step 6).")
    parser.add_argument("--split", default="all", choices=["train", "validation", "test", "all"])
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]

    products = pd.read_parquet(PRODUCTS_PATH)
    judgments = pd.read_parquet(JUDGMENTS_PATH)

    t0 = time.time()
    bm25_retriever, bm25_product_ids, _, _ = build_bm25_index(products)
    print(f"BM25 index built in {time.time()-t0:.1f}s")

    t0 = time.time()
    dense_index, dense_product_ids = load_dense_index()
    query_embeddings, query_emb_ids = load_query_embeddings()
    print(f"dense index + query embeddings loaded in {time.time()-t0:.1f}s")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    summaries = []
    for split in splits:
        t0 = time.time()
        candidates, queries_df, group_sizes = build_candidates_for_split(
            split, judgments, bm25_retriever, bm25_product_ids,
            dense_index, dense_product_ids, query_embeddings, query_emb_ids, k=args.k,
        )
        candidates.to_parquet(os.path.join(ARTIFACTS_DIR, f"candidates_{split}.parquet"), index=False)
        summary = summarize(split, candidates, queries_df, group_sizes)
        summary["build_time_s"] = time.time() - t0
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    with open(os.path.join(REPO_ROOT, "reports", "candidates_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
