import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import faiss
import lightgbm as lgb
import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval import ARTIFACTS_DIR, JUDGMENTS_PATH, PRODUCTS_PATH, build_bm25_index, retrieve_top_k
from candidates import _to_long, dense_retrieve, load_dense_index
from feature_extraction import build_title_bm25_index, build_features_for_chunk

MODEL_NAME = "BAAI/bge-base-en-v1.5"
LGBM_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "models", "lambdamart.txt")
K = 100

FEATURE_COLS = [
    "bm25_title_score", "bm25_body_score", "dense_cosine_similarity",
    "exact_matched_token_count", "query_token_coverage", "jaccard_similarity",
    "fuzzy_token_sort_ratio", "query_length", "product_title_length", "length_ratio",
    "brand_exact_match", "category_token_overlap", "negation_conflict_flag", "color_match_flag",
]

EXAMPLE_QUERIES = [
    "waterproof shoes",
    "10gal white trash can without lid",
    "pot container no smell",
    "coaching confidence",
    "tach adapter",
]


@st.cache_resource(show_spinner="Loading products, indexes, and trained model (one-time, ~1-2 min)...")
def load_everything():
    products = pd.read_parquet(PRODUCTS_PATH)
    judgments = pd.read_parquet(JUDGMENTS_PATH)

    bm25_retriever, bm25_product_ids, _, _ = build_bm25_index(products)
    title_retriever, title_product_ids = build_title_bm25_index(products)

    dense_index, dense_product_ids = load_dense_index()
    embed_model = SentenceTransformer(MODEL_NAME)

    lgbm_model = lgb.Booster(model_file=LGBM_MODEL_PATH)

    rewrites_path = os.path.join(os.path.dirname(__file__), "manual_gpt", "query_rewrites.csv")
    rewrites = pd.read_csv(rewrites_path) if os.path.exists(rewrites_path) else pd.DataFrame()

    title_lookup = dict(zip(products["product_id"], products["product_title"]))
    brand_lookup = dict(zip(products["product_id"], products["product_brand"]))

    return {
        "products": products, "judgments": judgments,
        "bm25_retriever": bm25_retriever, "bm25_product_ids": bm25_product_ids,
        "title_retriever": title_retriever, "title_product_ids": title_product_ids,
        "dense_index": dense_index, "dense_product_ids": dense_product_ids,
        "embed_model": embed_model, "lgbm_model": lgbm_model,
        "rewrites": rewrites, "title_lookup": title_lookup, "brand_lookup": brand_lookup,
    }


def run_query(query_text, resources, query_id=None):
    products = resources["products"]
    judgments = resources["judgments"]

    bm25_ids, bm25_scores = retrieve_top_k(resources["bm25_retriever"], resources["bm25_product_ids"], [query_text], K)
    q_emb = resources["embed_model"].encode([query_text])
    dense_ids, dense_scores = dense_retrieve(resources["dense_index"], resources["dense_product_ids"], q_emb, K)

    qid = query_id if query_id is not None else -1
    bm25_long = _to_long(bm25_ids, bm25_scores, np.array([qid]), "bm25_score", "from_bm25")
    dense_long = _to_long(dense_ids, dense_scores, np.array([qid]), "dense_score", "from_dense")
    candidates = pd.merge(bm25_long, dense_long, on=["query_id", "product_id"], how="outer")
    candidates["from_bm25"] = candidates["from_bm25"].fillna(False)
    candidates["from_dense"] = candidates["from_dense"].fillna(False)

    if query_id is not None:
        relevant = judgments[judgments["query_id"] == query_id][["query_id", "product_id", "relevance"]]
        candidates = candidates.merge(relevant, on=["query_id", "product_id"], how="left")
    else:
        candidates["relevance"] = np.nan
    candidates["is_judged"] = candidates["relevance"].notna()

    queries_df = pd.DataFrame({"query_id": [qid], "query": [query_text]})
    features, _ = build_features_for_chunk(
        candidates, queries_df, products, resources["title_retriever"], resources["title_product_ids"]
    )
    features["lgbm_score"] = resources["lgbm_model"].predict(features[FEATURE_COLS])
    features["product_title"] = features["product_id"].map(resources["title_lookup"])
    features["product_brand"] = features["product_id"].map(resources["brand_lookup"])
    return features


def display_ranking(df, score_col, title_lookup, n=10):
    top = df.sort_values(score_col, ascending=False).head(n).copy()
    top["title"] = top["product_id"].map(title_lookup)
    show_cols = ["title", "relevance", score_col]
    top["relevance"] = top["relevance"].fillna("unjudged")
    st.dataframe(top[show_cols].reset_index(drop=True), use_container_width=True)


def main():
    st.set_page_config(page_title="RankLab Demo", layout="wide")
    st.title("RankLab — Search Ranking Demo")
    st.caption(
        "Offline, local search-ranking pipeline (BM25 + dense retrieval + LambdaMART) over the Amazon ESCI "
        "dataset. No API calls in this demo's retrieval/ranking path — everything below runs locally against "
        "already-built indexes and the already-trained model."
    )

    resources = load_everything()

    col1, col2 = st.columns([3, 1])
    with col1:
        choice = st.selectbox("Pick an example query, or choose Custom below", ["Custom..."] + EXAMPLE_QUERIES)
    with col2:
        st.write("")

    if choice == "Custom...":
        query_text = st.text_input("Type a query", value="waterproof shoes")
    else:
        query_text = choice

    query_id = None
    matched_row = resources["judgments"][resources["judgments"]["query"] == query_text]
    if len(matched_row) > 0:
        query_id = int(matched_row["query_id"].iloc[0])
        st.caption(f"Matched dataset query_id={query_id} — ESCI relevance labels available for scoring context below.")

    if query_text.strip():
        with st.spinner("Retrieving + scoring..."):
            features = run_query(query_text, resources, query_id=query_id)

        tab_bm25, tab_dense, tab_lgbm, tab_features = st.tabs(["BM25 top-10", "Dense top-10", "LambdaMART top-10", "Feature detail (LambdaMART top-10)"])

        with tab_bm25:
            display_ranking(features, "bm25_body_score", resources["title_lookup"])
        with tab_dense:
            display_ranking(features, "dense_cosine_similarity", resources["title_lookup"])
        with tab_lgbm:
            display_ranking(features, "lgbm_score", resources["title_lookup"])
        with tab_features:
            top10 = features.sort_values("lgbm_score", ascending=False).head(10).copy()
            top10["title"] = top10["product_id"].map(resources["title_lookup"])
            st.dataframe(
                top10[["title", "lgbm_score"] + FEATURE_COLS].reset_index(drop=True),
                use_container_width=True,
            )

        rewrites = resources["rewrites"]
        if query_id is not None and len(rewrites) > 0:
            rw = rewrites[rewrites["query_id"] == query_id]
            if len(rw) > 0:
                row = rw.iloc[0]
                st.info(
                    f"**Step 9 query-rewrite study** — this query was in the 150-query sample.\n\n"
                    f"Raw: `{row['raw_query']}` -> Reviewed: `{row['reviewed_query']}`\n\n"
                    f"Source: **{row['model_label']}** (automated generation, human-reviewed — "
                    f"not a live/manual GPT chat interface; see README for the full methodology note). "
                    f"Review decision: **{row['review_decision']}** — {row['review_reason']}"
                )


if __name__ == "__main__":
    main()
