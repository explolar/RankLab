import argparse
import gc
import json
import os
import re
import time

import bm25s
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from retrieval import ARTIFACTS_DIR, JUDGMENTS_PATH, PRODUCTS_PATH, REPO_ROOT
from candidates import _to_long

CHUNK_SIZE_QUERIES = 2500  # bounds peak memory regardless of split size (train crashed at 3.6M rows unchunked)

REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
NEGATION_CUES = ("without", "no", "non", "not")

FEATURE_DICTIONARY = {
    "bm25_title_score": {
        "definition": "BM25 score of the candidate against a title-only index, for the candidate's own query. NaN if the candidate isn't in that query's title-index top-100.",
        "range": "[0, inf), NaN for out-of-top-100 candidates",
        "missing_value_handling": "fillna(0) — absence from title-index top-100 means negligible title-lexical relevance, 0 is a reasonable floor.",
        "why_it_helps": "isolates lexical match strength on the highest-signal field (title) from the noisier combined bm25_body_score.",
    },
    "bm25_body_score": {
        "definition": "BM25 score against the combined product_text (title+brand+bullets+color) index — reused directly from the Step 6 candidate-generation retrieval score.",
        "range": "[0, inf), NaN for candidates that entered the pool only via dense retrieval",
        "missing_value_handling": "fillna(0) — dense-only candidates weren't scored by BM25 at all.",
        "why_it_helps": "captures lexical match across all product text, not just the title.",
    },
    "dense_cosine_similarity": {
        "definition": "Cosine similarity between query and product bge-base-en-v1.5 embeddings — reused directly from Step 6 (embeddings pre-normalized, inner product == cosine).",
        "range": "[-1, 1], NaN for candidates that entered the pool only via BM25",
        "missing_value_handling": "fillna(0) — treated as neutral/unrelated in embedding space for BM25-only candidates.",
        "why_it_helps": "captures semantic relevance BM25 structurally cannot (synonyms, negation-free rephrasing) — documented directly in this project's own Step 3/6 failure analysis.",
    },
    "exact_matched_token_count": {
        "definition": "Count of query tokens that appear verbatim (case-insensitive) in the candidate's product_text tokens.",
        "range": "[0, query_length]",
        "missing_value_handling": "never missing (0 if no overlap).",
        "why_it_helps": "simple, explainable lexical overlap signal independent of BM25's IDF weighting.",
    },
    "query_token_coverage": {
        "definition": "exact_matched_token_count / max(query_length, 1).",
        "range": "[0, 1]",
        "missing_value_handling": "denominator floored at 1 — a small number of queries are non-Latin-script/symbol-only and tokenize to 0 words under the regex tokenizer; floor avoids NaN without dropping these real queries.",
        "why_it_helps": "normalizes match count by query length, so short and long queries are comparable.",
    },
    "jaccard_similarity": {
        "definition": "|query_tokens ∩ product_tokens| / |query_tokens ∪ product_tokens|.",
        "range": "[0, 1]",
        "missing_value_handling": "never missing.",
        "why_it_helps": "penalizes candidates with lots of extra unrelated vocabulary, unlike raw overlap count.",
    },
    "fuzzy_token_sort_ratio": {
        "definition": "rapidfuzz token_sort_ratio between query and product_title (word-order-independent fuzzy string similarity), scaled to [0,1].",
        "range": "[0, 1]",
        "missing_value_handling": "never missing.",
        "why_it_helps": "catches near-matches (typos, minor wording differences, word reordering) exact/Jaccard matching miss.",
    },
    "query_length": {
        "definition": "token count of the query.",
        "range": "[1, inf)",
        "missing_value_handling": "never missing.",
        "why_it_helps": "short queries behave differently (per Step 1 EDA and Step 9's planned rewrite study) — lets the model learn query-length-dependent behavior.",
    },
    "product_title_length": {
        "definition": "token count of product_title.",
        "range": "[1, inf)",
        "missing_value_handling": "never missing.",
        "why_it_helps": "very long or very short titles may correlate with relevance patterns (e.g. keyword-stuffed listings).",
    },
    "length_ratio": {
        "definition": "query_length / max(product_title_length, 1).",
        "range": "(0, inf)",
        "missing_value_handling": "denominator floored at 1 — a small number of product titles are non-Latin-script/symbol-only and tokenize to 0 words under the regex tokenizer; floor avoids Inf without dropping these real products.",
        "why_it_helps": "captures query-to-title specificity mismatch independent of either raw length alone.",
    },
    "brand_exact_match": {
        "definition": "1 if product_brand (lowercased) appears as a substring of the query, else 0. 0 if brand is null.",
        "range": "{0, 1}",
        "missing_value_handling": "0 when product_brand is null (5.6% of catalogue per Step 1 EDA) — absence of a brand can't match.",
        "why_it_helps": "brand queries are common and high-precision when they match.",
    },
    "category_token_overlap": {
        "definition": "Substitute for the plan's product_type feature (no such column exists in this ESCI export, flagged at Step 1). Token overlap fraction between query and product_description+product_bullet_point.",
        "range": "[0, 1]",
        "missing_value_handling": "0 when both description and bullet_point are null (documented in Step 1: ~this happens for a small fraction of the catalogue).",
        "why_it_helps": "approximates category/attribute relevance using the closest available fields.",
    },
    "negation_conflict_flag": {
        "definition": "1 if the query contains a negation cue ('without'/'no'/'non'/'not') AND the token immediately following it appears in the candidate's product_text, else 0. Directly operationalizes Step 3's documented BM25 failure mode (negation queries retrieving products that contain the excluded feature).",
        "range": "{0, 1}",
        "missing_value_handling": "0 when query has no negation cue.",
        "why_it_helps": "gives LambdaMART an explicit signal to down-rank exactly the failure pattern found in manual review — a feature this project's own data justified, not a generic textbook one.",
    },
    "color_match_flag": {
        "definition": "1 if product_color (lowercased) appears as a substring of the query, else 0. 0 if color is null.",
        "range": "{0, 1}",
        "missing_value_handling": "0 when product_color is null (32.2% of catalogue per Step 1 EDA).",
        "why_it_helps": "color is a common, high-precision query attribute when present in both query and product.",
    },
}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def extract_negation_target(query_tokens):
    for i, tok in enumerate(query_tokens):
        if tok in NEGATION_CUES and i + 1 < len(query_tokens):
            return query_tokens[i + 1]
    return None


def build_title_bm25_index(products):
    title_texts = products["product_title"].tolist()
    corpus_tokens = bm25s.tokenize(title_texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    product_id_array = products["product_id"].to_numpy()
    return retriever, product_id_array


def compute_title_bm25_scores(retriever, product_id_array, queries_df, k=100):
    query_tokens = bm25s.tokenize(queries_df["query"].tolist(), stopwords="en", show_progress=False)
    idx, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
    predicted_ids = product_id_array[idx]
    return _to_long(predicted_ids, scores, queries_df["query_id"].to_numpy(), "bm25_title_score", "_has_title_score")


def build_features_for_chunk(chunk_candidates, chunk_queries_df, products, title_retriever, title_product_ids):
    title_scores_long = compute_title_bm25_scores(title_retriever, title_product_ids, chunk_queries_df)
    features = chunk_candidates.merge(title_scores_long.drop(columns="_has_title_score"), on=["query_id", "product_id"], how="left")

    features = features.merge(products[["product_id", "product_title", "product_description", "product_bullet_point", "product_brand", "product_color"]], on="product_id", how="left")
    features = features.merge(chunk_queries_df, on="query_id", how="left")

    query_tok_cache = {row["query_id"]: tokenize(row["query"]) for _, row in chunk_queries_df.iterrows()}

    q_tokens = features["query_id"].map(query_tok_cache)
    title_tokens = features["product_title"].apply(tokenize)
    desc_bullet_tokens = (features["product_description"].fillna("") + " " + features["product_bullet_point"].fillna("")).apply(tokenize)

    def overlap_count(a, b):
        return len(set(a) & set(b))

    def jaccard(a, b):
        sa, sb = set(a), set(b)
        union = sa | sb
        return len(sa & sb) / len(union) if union else 0.0

    features["exact_matched_token_count"] = [overlap_count(q, t) for q, t in zip(q_tokens, title_tokens)]
    features["query_length"] = q_tokens.apply(len)
    # clip(lower=1): a small number of queries/titles are non-Latin-script or symbol-only and
    # tokenize to 0 words under this regex tokenizer; floor avoids NaN/Inf while keeping the
    # raw query_length/product_title_length columns themselves truthful (can still be 0).
    features["query_token_coverage"] = features["exact_matched_token_count"] / features["query_length"].clip(lower=1)
    features["jaccard_similarity"] = [jaccard(q, t) for q, t in zip(q_tokens, title_tokens)]
    features["fuzzy_token_sort_ratio"] = [
        fuzz.token_sort_ratio(q, t) / 100.0
        for q, t in zip(features["query"], features["product_title"])
    ]
    features["product_title_length"] = title_tokens.apply(len)
    features["length_ratio"] = features["query_length"] / features["product_title_length"].clip(lower=1)

    brand_lower = features["product_brand"].str.lower()
    features["brand_exact_match"] = [
        int(pd.notna(b) and b in q.lower()) for b, q in zip(brand_lower, features["query"])
    ]

    features["category_token_overlap"] = [jaccard(q, t) for q, t in zip(q_tokens, desc_bullet_tokens)]

    negation_targets = {qid: extract_negation_target(toks) for qid, toks in query_tok_cache.items()}
    features["negation_conflict_flag"] = [
        int(negation_targets.get(qid) is not None and negation_targets[qid] in ptoks)
        for qid, ptoks in zip(features["query_id"], desc_bullet_tokens + title_tokens)
    ]

    color_lower = features["product_color"].str.lower()
    features["color_match_flag"] = [
        int(pd.notna(c) and c in q.lower()) for c, q in zip(color_lower, features["query"])
    ]

    features["bm25_title_score"] = features["bm25_title_score"].fillna(0.0)
    features["bm25_score"] = features["bm25_score"].fillna(0.0)
    features["dense_score"] = features["dense_score"].fillna(0.0)
    features = features.rename(columns={"bm25_score": "bm25_body_score", "dense_score": "dense_cosine_similarity"})

    feature_cols = [
        "bm25_title_score", "bm25_body_score", "dense_cosine_similarity",
        "exact_matched_token_count", "query_token_coverage", "jaccard_similarity",
        "fuzzy_token_sort_ratio", "query_length", "product_title_length", "length_ratio",
        "brand_exact_match", "category_token_overlap", "negation_conflict_flag", "color_match_flag",
    ]
    id_cols = ["query_id", "product_id", "relevance", "is_judged", "from_bm25", "from_dense"]
    result = features[id_cols + feature_cols].copy()

    del features, q_tokens, title_tokens, desc_bullet_tokens, query_tok_cache, negation_targets
    gc.collect()

    return result, feature_cols


def build_features_for_split(split, products, judgments, chunk_size=CHUNK_SIZE_QUERIES):
    candidates_path = os.path.join(ARTIFACTS_DIR, f"candidates_{split}.parquet")
    all_query_ids = pq.read_table(candidates_path, columns=["query_id"])["query_id"].to_pandas().unique()
    all_query_ids.sort()

    title_retriever, title_product_ids = build_title_bm25_index(products)

    out_path = os.path.join(ARTIFACTS_DIR, f"features_{split}.parquet")
    writer = None
    n_rows_total, n_nan_total, n_inf_total = 0, 0, 0
    feature_cols = None

    n_chunks = (len(all_query_ids) + chunk_size - 1) // chunk_size
    for c in range(n_chunks):
        chunk_qids = all_query_ids[c * chunk_size:(c + 1) * chunk_size].tolist()

        candidates_table = pq.read_table(candidates_path, filters=[("query_id", "in", chunk_qids)])
        chunk_candidates = candidates_table.to_pandas().reset_index(drop=True)
        del candidates_table

        chunk_queries_df = (
            judgments[judgments["query_id"].isin(chunk_qids)]
            .drop_duplicates("query_id")[["query_id", "query"]].reset_index(drop=True)
        )

        chunk_features, feature_cols = build_features_for_chunk(
            chunk_candidates, chunk_queries_df, products, title_retriever, title_product_ids
        )
        del chunk_candidates, chunk_queries_df

        table = pa.Table.from_pandas(chunk_features, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

        n_rows_total += len(chunk_features)
        n_nan_total += int(chunk_features[feature_cols].isna().sum().sum())
        n_inf_total += int(np.isinf(chunk_features[feature_cols].to_numpy(dtype=float)).sum())

        print(f"  chunk {c+1}/{n_chunks}: {len(chunk_features)} rows written")
        del chunk_features, table
        gc.collect()

    if writer is not None:
        writer.close()

    return {
        "split": split,
        "n_rows": n_rows_total,
        "n_features": len(feature_cols) if feature_cols else 0,
        "n_nan": n_nan_total,
        "n_inf": n_inf_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract the 14 ranking features (RankLab Step 7).")
    parser.add_argument("--split", default="all", choices=["train", "validation", "test", "all"])
    args = parser.parse_args()

    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]

    products = pd.read_parquet(PRODUCTS_PATH)
    judgments = pd.read_parquet(JUDGMENTS_PATH)

    for split in splits:
        t0 = time.time()
        summary = build_features_for_split(split, products, judgments)
        summary["build_time_s"] = time.time() - t0
        print(json.dumps(summary, indent=2))

    with open(os.path.join(REPORTS_DIR, "feature_dictionary.json"), "w") as f:
        json.dump(FEATURE_DICTIONARY, f, indent=2)
    print("saved reports/feature_dictionary.json")


if __name__ == "__main__":
    main()
