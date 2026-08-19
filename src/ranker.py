import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from retrieval import ARTIFACTS_DIR, JUDGMENTS_PATH, REPO_ROOT, dcg

REPORTS_DIR = os.path.join(REPO_ROOT, "reports")

FEATURE_COLS = [
    "bm25_title_score", "bm25_body_score", "dense_cosine_similarity",
    "exact_matched_token_count", "query_token_coverage", "jaccard_similarity",
    "fuzzy_token_sort_ratio", "query_length", "product_title_length", "length_ratio",
    "brand_exact_match", "category_token_overlap", "negation_conflict_flag", "color_match_flag",
]

# Plan Step 8.2: conservative starting parameters.
LGB_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    n_estimators=1000,
    random_state=42,
)


def load_features(split):
    return pd.read_parquet(os.path.join(ARTIFACTS_DIR, f"features_{split}.parquet"))


def make_groups(df):
    return df.groupby("query_id", sort=False).size().to_numpy()


def minmax_normalize_within_group(df, col):
    g = df.groupby("query_id")[col]
    mn = g.transform("min")
    mx = g.transform("max")
    rng = (mx - mn).replace(0, 1)
    return (df[col] - mn) / rng


def rank_by_score(df, score_col):
    ordered = {}
    for qid, g in df.groupby("query_id", sort=False):
        ordered[qid] = g.sort_values(score_col, ascending=False)["product_id"].tolist()
    return ordered


def compute_ranking_metrics(ordered_by_query, judgments_df, k=10):
    query_relevance = {
        qid: dict(zip(g["product_id"], g["relevance"]))
        for qid, g in judgments_df.groupby("query_id")
    }
    records = []
    for qid, preds in ordered_by_query.items():
        rel_lookup = query_relevance.get(qid, {})
        total_relevant = sum(1 for r in rel_lookup.values() if r >= 1)
        top_k = preds[:k]
        top_rel = [rel_lookup.get(pid, 0) for pid in top_k]
        retrieved_relevant = sum(1 for r in top_rel if r >= 1)
        recall_k = retrieved_relevant / total_relevant if total_relevant else 0.0

        ideal_rel = sorted(rel_lookup.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal_rel)
        ndcg_k = dcg(top_rel) / ideal_dcg if ideal_dcg > 0 else 0.0

        mrr_k = 0.0
        for rank, r in enumerate(top_rel, start=1):
            if r >= 1:
                mrr_k = 1.0 / rank
                break

        records.append({"query_id": qid, f"recall_{k}": recall_k, f"ndcg_{k}": ndcg_k, f"mrr_{k}": mrr_k})
    return pd.DataFrame(records)


def train_model(train_judged, val_judged, param_overrides=None):
    X_train = train_judged[FEATURE_COLS]
    y_train = train_judged["relevance"].astype(int)
    train_group = make_groups(train_judged)

    X_val = val_judged[FEATURE_COLS]
    y_val = val_judged["relevance"].astype(int)
    val_group = make_groups(val_judged)

    params = {**LGB_PARAMS, **(param_overrides or {})}
    model = lgb.LGBMRanker(**params)
    model.fit(
        X_train, y_train, group=train_group,
        eval_set=[(X_val, y_val)], eval_group=[val_group],
        eval_at=[10],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
    )
    return model


def evaluate_orderings(df, model, judgments, k=10):
    df = df.reset_index(drop=True).copy()
    df["union_score"] = (
        minmax_normalize_within_group(df, "bm25_body_score")
        + minmax_normalize_within_group(df, "dense_cosine_similarity")
    )
    df["lgbm_score"] = model.predict(df[FEATURE_COLS])

    orderings = {
        "bm25": "bm25_body_score",
        "dense": "dense_cosine_similarity",
        "candidate_union": "union_score",
        "lambdamart": "lgbm_score",
    }

    results = {}
    per_query = {}
    for name, score_col in orderings.items():
        ordered = rank_by_score(df, score_col)
        metrics_df = compute_ranking_metrics(ordered, judgments, k=k)
        results[name] = {
            f"recall_{k}_mean": float(metrics_df[f"recall_{k}"].mean()),
            f"ndcg_{k}_mean": float(metrics_df[f"ndcg_{k}"].mean()),
            f"mrr_{k}_mean": float(metrics_df[f"mrr_{k}"].mean()),
        }
        per_query[name] = metrics_df
    return results, per_query, df


# Plan Step 8.3: "tune only a small grid: tree complexity, learning rate, minimum leaf
# size, and regularization." Kept intentionally small (6 configs, one param varied at a
# time from the conservative baseline), not an exhaustive search.
TUNING_GRID = [
    {"label": "baseline", "overrides": {}},
    {"label": "num_leaves=15", "overrides": {"num_leaves": 15}},
    {"label": "num_leaves=63", "overrides": {"num_leaves": 63}},
    {"label": "learning_rate=0.1", "overrides": {"learning_rate": 0.1}},
    {"label": "min_child_samples=50", "overrides": {"min_child_samples": 50}},
    {"label": "reg_lambda=1.0", "overrides": {"reg_lambda": 1.0}},
]


def run_grid_search(train_judged, val_judged):
    results = []
    for cfg in TUNING_GRID:
        t0 = time.time()
        model = train_model(train_judged, val_judged, param_overrides=cfg["overrides"])
        elapsed = time.time() - t0
        best_score = model.best_score_["valid_0"]["ndcg@10"]
        results.append({"label": cfg["label"], "overrides": cfg["overrides"], "val_ndcg_10": best_score, "best_iteration": int(model.best_iteration_), "train_time_s": elapsed})
        print(f"  {cfg['label']}: val_ndcg@10={best_score:.4f} (best_iter={model.best_iteration_}, {elapsed:.1f}s)")
    return sorted(results, key=lambda r: -r["val_ndcg_10"])


def main():
    parser = argparse.ArgumentParser(description="Train + evaluate LambdaMART (RankLab Step 8).")
    parser.add_argument("--evaluate-test", action="store_true", help="also evaluate the frozen model on the test split (run once only)")
    parser.add_argument("--grid-search", action="store_true", help="run the small tuning grid (plan Step 8.3) and use the best config")
    args = parser.parse_args()

    train_df = load_features("train")
    val_df = load_features("validation")
    judgments = pd.read_parquet(JUDGMENTS_PATH)

    train_judged = train_df[train_df["is_judged"]].reset_index(drop=True)
    val_judged = val_df[val_df["is_judged"]].reset_index(drop=True)
    print(f"train judged rows: {len(train_judged)}, groups: {len(make_groups(train_judged))}")
    print(f"val judged rows: {len(val_judged)}, groups: {len(make_groups(val_judged))}")

    best_overrides = {}
    if args.grid_search:
        print("running tuning grid...")
        grid_results = run_grid_search(train_judged, val_judged)
        best_overrides = grid_results[0]["overrides"]
        print(f"best config: {grid_results[0]['label']} (val_ndcg@10={grid_results[0]['val_ndcg_10']:.4f})")
        with open(os.path.join(REPORTS_DIR, "ranker_tuning_grid.json"), "w") as f:
            json.dump(grid_results, f, indent=2)

    t0 = time.time()
    model = train_model(train_judged, val_judged, param_overrides=best_overrides)
    train_time = time.time() - t0
    print(f"trained in {train_time:.1f}s, best_iteration={model.best_iteration_}")

    val_judgments = judgments[judgments["query_id"].isin(val_df["query_id"].unique())]
    results, per_query, val_scored = evaluate_orderings(val_df, model, val_judgments, k=10)
    print(json.dumps(results, indent=2))

    importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    print("feature importance (gain-split count):", json.dumps(importance, indent=2))

    sample = val_judged.sample(min(5000, len(val_judged)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[FEATURE_COLS])
    mean_abs_shap = dict(zip(FEATURE_COLS, np.abs(shap_values).mean(axis=0).tolist()))
    mean_abs_shap = dict(sorted(mean_abs_shap.items(), key=lambda x: -x[1]))

    os.makedirs(REPORTS_DIR, exist_ok=True)
    summary = {
        "train_rows": len(train_judged),
        "val_rows_judged": len(val_judged),
        "train_time_s": train_time,
        "best_iteration": int(model.best_iteration_),
        "grid_search_run": args.grid_search,
        "chosen_param_overrides": best_overrides,
        "comparison": results,
        "feature_importance_gain": importance,
        "mean_abs_shap": mean_abs_shap,
    }
    with open(os.path.join(REPORTS_DIR, "ranker_validation_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for name, mdf in per_query.items():
        mdf.to_parquet(os.path.join(ARTIFACTS_DIR, f"ranker_perquery_{name}.parquet"), index=False)

    model.booster_.save_model(os.path.join(ARTIFACTS_DIR, "models", "lambdamart.txt"))
    print("saved model, per-query metrics, and reports/ranker_validation_results.json")

    if args.evaluate_test:
        test_df = load_features("test")
        test_judgments = judgments[judgments["query_id"].isin(test_df["query_id"].unique())]
        test_results, test_per_query, _ = evaluate_orderings(test_df, model, test_judgments, k=10)
        print("TEST RESULTS (final, single evaluation):")
        print(json.dumps(test_results, indent=2))
        with open(os.path.join(REPORTS_DIR, "ranker_test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)
        for name, mdf in test_per_query.items():
            mdf.to_parquet(os.path.join(ARTIFACTS_DIR, f"ranker_perquery_test_{name}.parquet"), index=False)


if __name__ == "__main__":
    main()
