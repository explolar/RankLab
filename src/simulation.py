import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from retrieval import ARTIFACTS_DIR, CONFIG_PATH, REPO_ROOT

REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
ARMS_PATH = os.path.join(ARTIFACTS_DIR, "experiment_arms.parquet")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["simulation"]


def load_arm_matrices():
    """Returns {arm: (n_queries, top_n) relevance matrix}, query_ids list, qid->row-index map."""
    df = pd.read_parquet(ARMS_PATH)
    query_ids = sorted(df["query_id"].unique())
    qid_to_idx = {q: i for i, q in enumerate(query_ids)}
    top_n = int(df["rank"].max())

    matrices = {}
    for arm, g in df.groupby("arm"):
        mat = np.zeros((len(query_ids), top_n))
        rows = g["query_id"].map(qid_to_idx).to_numpy()
        cols = (g["rank"].to_numpy() - 1).astype(int)
        mat[rows, cols] = g["relevance"].to_numpy()
        matrices[arm] = mat
    return matrices, query_ids, qid_to_idx


def examine_probs(top_n):
    ranks = np.arange(1, top_n + 1)
    return 1 / np.log2(ranks + 1)


def simulate_clicks(mat, sampled_idx, user_mult, attract_arr, ex_probs, rng):
    """Independent-per-position click model (plan Step 11.4): each of the top_n
    positions gets its own Bernoulli(examine_i * attract_i * user_mult) draw.
    Session outcome Y = 1 if any position was clicked. Not a cascade model."""
    rel = mat[sampled_idx].astype(int)
    attract = attract_arr[rel]
    click_prob = np.clip(ex_probs[None, :] * attract * user_mult[:, None], 0, 1)
    clicks = rng.random(click_prob.shape) < click_prob
    return clicks.any(axis=1).astype(float)


def run_one_seed(matrices, query_ids, cfg, seed):
    rng = np.random.default_rng(seed)
    top_n = cfg["top_n"]
    ex_probs = examine_probs(top_n)
    attract_arr = np.array([cfg["attraction_probability"][g] for g in range(4)])
    n_users = cfg["users_per_arm"]
    user_effect_std = cfg["user_effect_std"]
    pre_arm = cfg["pre_period_arm"]

    records = {}
    for arm in cfg["arms"]:
        sampled_idx = rng.integers(0, len(query_ids), size=n_users)
        user_mult = np.clip(rng.normal(1.0, user_effect_std, size=n_users), 0.2, 2.5)

        Y_exp = simulate_clicks(matrices[arm], sampled_idx, user_mult, attract_arr, ex_probs, rng)
        # Pre-period: SAME simulated users (same sampled query + same latent user_mult)
        # experience the pre-period baseline arm's relevance sequence -- this shared
        # user_mult is what correlates Y_pre with Y_exp (plan Step 11.5), independent
        # click draw so pre and exp aren't literally identical even when arm == pre_arm.
        Y_pre = simulate_clicks(matrices[pre_arm], sampled_idx, user_mult, attract_arr, ex_probs, rng)

        records[arm] = pd.DataFrame({"Y_exp": Y_exp, "Y_pre": Y_pre})
    return records


def compute_cuped_theta(all_Y_exp, all_Y_pre):
    """Theta estimated once, pooled across all arms (plan Step 12.3: 'only from
    appropriate data' -- Y_pre is generated identically regardless of arm assignment,
    so it isn't affected by treatment; pooling is the standard CUPED approach)."""
    var_pre = np.var(all_Y_pre, ddof=1)
    if var_pre == 0:
        return 0.0
    cov = np.cov(all_Y_exp, all_Y_pre)[0, 1]
    return cov / var_pre


def apply_cuped(Y_exp, Y_pre, theta, pre_mean):
    return Y_exp - theta * (Y_pre - pre_mean)


def two_proportion_test(Y1, Y2):
    n1, n2 = len(Y1), len(Y2)
    p1, p2 = float(Y1.mean()), float(Y2.mean())
    p_pool = (Y1.sum() + Y2.sum()) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {
        "lift": p1 - p2, "p1": p1, "p2": p2, "z": float(z), "p_value": float(p_value),
        "ci_95_low": float(p1 - p2 - 1.96 * se_diff), "ci_95_high": float(p1 - p2 + 1.96 * se_diff),
    }


def obf_boundary(info_fraction, z_final):
    """Approximate O'Brien-Fleming boundary: z_final / sqrt(t). A standard textbook
    approximation, not a numerically-solved alpha-spending function (documented here
    since the plan asks for O'Brien-Fleming without specifying implementation depth)."""
    return z_final / np.sqrt(info_fraction)


def sample_size_two_proportion(p_baseline, mde, alpha, power):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p2 = p_baseline + mde
    p_bar = (p_baseline + p2) / 2
    n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(p_baseline * (1 - p_baseline) + p2 * (1 - p2))) ** 2) / (mde ** 2)
    return int(np.ceil(n))


def benjamini_hochberg(p_values, q=0.05):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    adjusted = np.empty(n)
    adjusted[order] = adjusted_ranked
    return adjusted


def analyze_seed(records, cfg):
    arms = cfg["arms"]
    baseline = arms[0]
    comparisons = [(arms[1], baseline), (arms[2], baseline), (arms[1], arms[2])]

    all_Y_exp = np.concatenate([records[a]["Y_exp"].to_numpy() for a in arms])
    all_Y_pre = np.concatenate([records[a]["Y_pre"].to_numpy() for a in arms])
    theta = compute_cuped_theta(all_Y_exp, all_Y_pre)
    pre_mean = all_Y_pre.mean()

    results = {}
    raw_p_values = []
    keys = []
    for arm1, arm2 in comparisons:
        Y1, Y2 = records[arm1]["Y_exp"].to_numpy(), records[arm2]["Y_exp"].to_numpy()
        test = two_proportion_test(Y1, Y2)

        Y1_cuped = apply_cuped(Y1, records[arm1]["Y_pre"].to_numpy(), theta, pre_mean)
        Y2_cuped = apply_cuped(Y2, records[arm2]["Y_pre"].to_numpy(), theta, pre_mean)
        var_before = float(np.var(np.concatenate([Y1, Y2]), ddof=1))
        var_after = float(np.var(np.concatenate([Y1_cuped, Y2_cuped]), ddof=1))
        var_reduction = 1 - var_after / var_before if var_before > 0 else 0.0

        key = f"{arm1}_vs_{arm2}"
        keys.append(key)
        raw_p_values.append(test["p_value"])
        results[key] = {**test, "var_before": var_before, "var_after_cuped": var_after, "cuped_variance_reduction": var_reduction}

    adjusted_p = benjamini_hochberg(raw_p_values, cfg["fdr_q"])
    for key, adj in zip(keys, adjusted_p):
        results[key]["p_value_bh_adjusted"] = float(adj)

    return results, float(theta)


def sequential_analysis(records, cfg):
    z_final = stats.norm.ppf(1 - cfg["alpha"] / 2)
    arms = cfg["arms"]
    baseline = arms[0]
    n_users = len(records[baseline])

    seq_results = []
    for frac in cfg["info_fractions"]:
        n_at_frac = max(1, int(round(n_users * frac)))
        boundary = obf_boundary(frac, z_final)
        row = {"info_fraction": frac, "n_per_arm": n_at_frac, "obf_boundary_z": float(boundary)}
        for arm in arms[1:]:
            Y_treat = records[arm]["Y_exp"].to_numpy()[:n_at_frac]
            Y_base = records[baseline]["Y_exp"].to_numpy()[:n_at_frac]
            test = two_proportion_test(Y_treat, Y_base)
            row[f"{arm}_vs_{baseline}_z"] = test["z"]
            row[f"{arm}_vs_{baseline}_unadjusted_p"] = test["p_value"]
            row[f"{arm}_vs_{baseline}_boundary_crossed"] = bool(abs(test["z"]) > boundary)
        seq_results.append(row)
    return seq_results


def stability_analysis(matrices, query_ids, cfg, n_seeds):
    arms = cfg["arms"]
    baseline = arms[0]
    win_counts = {a: 0 for a in arms}
    lifts = {a: [] for a in arms[1:]}
    var_reductions = {a: [] for a in arms[1:]}
    bh_significant = {a: 0 for a in arms[1:]}

    for seed in range(n_seeds):
        records = run_one_seed(matrices, query_ids, cfg, seed)
        results, _ = analyze_seed(records, cfg)

        means = {a: records[a]["Y_exp"].mean() for a in arms}
        win_counts[max(means, key=means.get)] += 1

        for arm in arms[1:]:
            key = f"{arm}_vs_{baseline}"
            lifts[arm].append(results[key]["lift"])
            var_reductions[arm].append(results[key]["cuped_variance_reduction"])
            if results[key]["p_value_bh_adjusted"] < cfg["fdr_q"]:
                bh_significant[arm] += 1

    return {
        "n_seeds": n_seeds,
        "share_wins": {a: win_counts[a] / n_seeds for a in arms},
        "avg_lift_vs_baseline": {a: float(np.mean(lifts[a])) for a in lifts},
        "std_lift_vs_baseline": {a: float(np.std(lifts[a])) for a in lifts},
        "avg_cuped_variance_reduction": {a: float(np.mean(var_reductions[a])) for a in var_reductions},
        "share_bh_significant_vs_baseline": {a: bh_significant[a] / n_seeds for a in bh_significant},
    }


def main():
    parser = argparse.ArgumentParser(description="Synthetic click experiment simulator (RankLab Step 11/12).")
    parser.add_argument("--stability-seeds", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    matrices, query_ids, _ = load_arm_matrices()

    print("=== DEBUG RUN (seed={}) ===".format(cfg["debug_seed"]))
    debug_records = run_one_seed(matrices, query_ids, cfg, cfg["debug_seed"])
    ctr_by_arm = {a: float(debug_records[a]["Y_exp"].mean()) for a in cfg["arms"]}
    print("CTR by arm:", ctr_by_arm)

    arms = cfg["arms"]
    sanity_ok = ctr_by_arm[arms[0]] <= ctr_by_arm[arms[1]]
    print(f"sanity checkpoint (stronger-relevance arm {arms[1]} >= weaker-relevance arm {arms[0]} CTR):", sanity_ok)

    results, theta = analyze_seed(debug_records, cfg)
    print("\npairwise comparisons:")
    print(json.dumps(results, indent=2))
    print("\nCUPED theta (pooled):", theta)

    seq = sequential_analysis(debug_records, cfg)
    print("\nsequential (O'Brien-Fleming) analysis:")
    print(json.dumps(seq, indent=2))

    power_n = sample_size_two_proportion(
        cfg["planning_baseline_ctr"], cfg["mde_ctr_pp"], cfg["alpha"], cfg["power"]
    )
    print(f"\nrequired sample size per arm (planning assumption baseline_ctr={cfg['planning_baseline_ctr']}, "
          f"mde={cfg['mde_ctr_pp']}, alpha={cfg['alpha']}, power={cfg['power']}): {power_n}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    debug_summary = {
        "seed": cfg["debug_seed"],
        "ctr_by_arm": ctr_by_arm,
        "sanity_checkpoint_passed": bool(sanity_ok),
        "cuped_theta": theta,
        "pairwise_comparisons": results,
        "sequential_analysis": seq,
        "required_sample_size_per_arm": power_n,
        "config": cfg,
    }
    with open(os.path.join(REPORTS_DIR, "simulation_debug_run.json"), "w") as f:
        json.dump(debug_summary, f, indent=2)

    n_seeds = args.stability_seeds or cfg["stability_seeds"]
    print(f"\n=== STABILITY ANALYSIS ({n_seeds} seeds) ===")
    stability = stability_analysis(matrices, query_ids, cfg, n_seeds)
    print(json.dumps(stability, indent=2))

    with open(os.path.join(REPORTS_DIR, "simulation_stability.json"), "w") as f:
        json.dump(stability, f, indent=2)

    print("\nsaved reports/simulation_debug_run.json and reports/simulation_stability.json")


if __name__ == "__main__":
    main()
