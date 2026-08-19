import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from retrieval import ARTIFACTS_DIR, JUDGMENTS_PATH, PRODUCTS_PATH, REPO_ROOT, build_bm25_index, retrieve_top_k
from candidates import _to_long, dense_retrieve, load_dense_index
from feature_extraction import build_title_bm25_index, build_features_for_chunk

REWRITES_PATH = os.path.join(REPO_ROOT, "manual_gpt", "query_rewrites.csv")
MODEL_NAME = "BAAI/bge-base-en-v1.5"
LGBM_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "models", "lambdamart.txt")
OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "experiment_arms.parquet")
K = 100
TOP_N = 10

FEATURE_COLS = [
    "bm25_title_score", "bm25_body_score", "dense_cosine_similarity",
    "exact_matched_token_count", "query_token_coverage", "jaccard_similarity",
    "fuzzy_token_sort_ratio", "query_length", "product_title_length", "length_ratio",
    "brand_exact_match", "category_token_overlap", "negation_conflict_flag", "color_match_flag",
]


def top_n_relevance(df, score_col, n=TOP_N):
    """For each query_id, sort by score_col desc, return relevance at ranks 1..n (0-padded if fewer candidates)."""
    records = []
    for qid, g in df.groupby("query_id"):
        top = g.sort_values(score_col, ascending=False).head(n)
        rel = top["relevance"].fillna(0).to_numpy(dtype=float)
        if len(rel) < n:
            rel = np.pad(rel, (0, n - len(rel)))
        for rank, r in enumerate(rel, start=1):
            records.append({"query_id": qid, "rank": rank, "relevance": r})
    return pd.DataFrame(records)


def build_arm_c(rewrites, products, judgments, title_retriever, title_product_ids, lgbm_model):
    query_id_array = rewrites["query_id"].to_numpy()
    reviewed_texts = rewrites["reviewed_query"].tolist()

    print("  BM25 retrieval on reviewed_query text...")
    bm25_retriever, bm25_product_ids, _, _ = build_bm25_index(products)
    bm25_ids, bm25_scores = retrieve_top_k(bm25_retriever, bm25_product_ids, reviewed_texts, K)

    print("  dense retrieval on reviewed_query text...")
    dense_index, dense_product_ids = load_dense_index()
    model = SentenceTransformer(MODEL_NAME)
    reviewed_emb = model.encode(reviewed_texts, batch_size=32, show_progress_bar=False)
    dense_ids, dense_scores = dense_retrieve(dense_index, dense_product_ids, reviewed_emb, K)

    bm25_long = _to_long(bm25_ids, bm25_scores, query_id_array, "bm25_score", "from_bm25")
    dense_long = _to_long(dense_ids, dense_scores, query_id_array, "dense_score", "from_dense")
    candidates = pd.merge(bm25_long, dense_long, on=["query_id", "product_id"], how="outer")
    candidates["from_bm25"] = candidates["from_bm25"].fillna(False)
    candidates["from_dense"] = candidates["from_dense"].fillna(False)

    relevant_judgments = judgments[judgments["query_id"].isin(query_id_array)]
    candidates = candidates.merge(
        relevant_judgments[["query_id", "product_id", "relevance"]], on=["query_id", "product_id"], how="left"
    )
    candidates["is_judged"] = candidates["relevance"].notna()

    queries_df = rewrites[["query_id", "reviewed_query"]].rename(columns={"reviewed_query": "query"})

    print("  extracting features...")
    features, _ = build_features_for_chunk(candidates, queries_df, products, title_retriever, title_product_ids)
    features["lgbm_score"] = lgbm_model.predict(features[FEATURE_COLS])
    return features


def main():
    rewrites = pd.read_csv(REWRITES_PATH)
    products = pd.read_parquet(PRODUCTS_PATH)
    judgments = pd.read_parquet(JUDGMENTS_PATH)
    query_ids = rewrites["query_id"].tolist()

    print("Arm A/B: loading features_test.parquet for the 150 rewrite-study queries...")
    features_test = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "features_test.parquet"))
    subset = features_test[features_test["query_id"].isin(query_ids)].copy()

    lgbm_model = lgb.Booster(model_file=LGBM_MODEL_PATH)
    subset["lgbm_score"] = lgbm_model.predict(subset[FEATURE_COLS])

    arm_a = top_n_relevance(subset, "bm25_body_score")
    arm_a["arm"] = "A_bm25"

    arm_b = top_n_relevance(subset, "lgbm_score")
    arm_b["arm"] = "B_lambdamart"

    print("Arm C: rebuilding candidates/features/scores on reviewed_query text...")
    t0 = time.time()
    title_retriever, title_product_ids = build_title_bm25_index(products)
    arm_c_features = build_arm_c(rewrites, products, judgments, title_retriever, title_product_ids, lgbm_model)
    print(f"  done in {time.time()-t0:.1f}s")

    arm_c = top_n_relevance(arm_c_features, "lgbm_score")
    arm_c["arm"] = "C_lambdamart_rewrite"

    combined = pd.concat([arm_a, arm_b, arm_c], ignore_index=True)
    combined.to_parquet(OUTPUT_PATH, index=False)

    print("\nsaved", OUTPUT_PATH, ":", len(combined), "rows")
    summary = combined.groupby("arm")["relevance"].agg(["mean", "count"])
    print(summary)

    # sanity: mean relevance at rank 1 per arm (should broadly increase A -> B -> C if pipeline is working)
    rank1 = combined[combined["rank"] == 1].groupby("arm")["relevance"].mean()
    print("\nmean relevance at rank 1, per arm:")
    print(rank1)


if __name__ == "__main__":
    main()
