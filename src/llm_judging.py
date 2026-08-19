import argparse
import csv
import os
import time
from datetime import datetime, timezone

import pandas as pd

from retrieval import REPO_ROOT
from llm_client import DEFAULT_MODEL, call_groq_with_ratelimit, parse_json_response

TOKEN_SAFETY_BUFFER = 1000  # proactively pause before hitting 0 remaining, rather than reacting to a 429

MANUAL_GPT_DIR = os.path.join(REPO_ROOT, "manual_gpt")
PROMPT_PATH = os.path.join(MANUAL_GPT_DIR, "prompts", "judging_v1.md")
SAMPLE_PATH = os.path.join(MANUAL_GPT_DIR, "judging_study_blinded.csv")
OUTPUT_PATH = os.path.join(MANUAL_GPT_DIR, "judging_study_results.csv")

FIELDNAMES = ["pair_id", "query_id", "product_id", "raw_response", "gpt_score", "rationale", "parse_status", "model_label", "run_date"]


def load_prompt_template():
    with open(PROMPT_PATH) as f:
        return f.read()


def build_prompt(template, row):
    return (
        template.replace("{QUERY}", str(row["query"]))
        .replace("{TITLE}", str(row["title"]))
        .replace("{BRAND}", str(row["brand"]) if pd.notna(row["brand"]) else "unknown")
        .replace("{PRODUCT_TEXT}", str(row["product_text"]) if pd.notna(row["product_text"]) else "")
    )


def load_completed_pair_ids():
    if not os.path.exists(OUTPUT_PATH):
        return set()
    existing = pd.read_csv(OUTPUT_PATH)
    return set(existing["pair_id"])


def main():
    parser = argparse.ArgumentParser(description="Automated (Groq API) relevance judging for the Step 10 sample.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N pairs (for a quick test)")
    parser.add_argument("--delay", type=float, default=0.3, help="seconds to sleep between requests, to stay under rate limits")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY environment variable not set.")

    prompt_template = load_prompt_template()
    sample = pd.read_csv(SAMPLE_PATH)
    if args.limit:
        sample = sample.head(args.limit)

    completed = load_completed_pair_ids()
    if completed:
        print(f"resuming: {len(completed)} pairs already done, skipping those")
        sample = sample[~sample["pair_id"].isin(completed)]

    file_exists = os.path.exists(OUTPUT_PATH)
    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for i, row in sample.iterrows():
            prompt_text = build_prompt(prompt_template, row)
            raw_response, ratelimit = call_groq_with_ratelimit(prompt_text, api_key, args.model)
            parsed, status = parse_json_response(raw_response)

            record = {
                "pair_id": row["pair_id"],
                "query_id": row["query_id"],
                "product_id": row["product_id"],
                "raw_response": raw_response,
                "gpt_score": parsed.get("score", None),
                "rationale": parsed.get("rationale", ""),
                "parse_status": status,
                "model_label": f"{args.model} (Groq API, automated)",
                "run_date": datetime.now(timezone.utc).isoformat(),
            }
            writer.writerow(record)
            f.flush()

            remaining = ratelimit["remaining_tokens"]
            if remaining is not None and remaining < TOKEN_SAFETY_BUFFER:
                wait = ratelimit["reset_tokens_s"] + 0.5
                print(f"[{row['pair_id']}] score={parsed.get('score', '?')} ({status}) -- low on tokens ({remaining} left), pausing {wait:.1f}s")
                time.sleep(wait)
            else:
                print(f"[{row['pair_id']}] score={parsed.get('score', '?')} ({status}) [{remaining} tokens left]")
                time.sleep(args.delay)

    final = pd.read_csv(OUTPUT_PATH)
    print(f"\ntotal saved to {OUTPUT_PATH}: {len(final)} rows")
    print(f"parse_status counts: {final['parse_status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
