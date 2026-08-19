import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from retrieval import REPO_ROOT
from llm_client import DEFAULT_MODEL, call_groq, parse_json_response

MANUAL_GPT_DIR = os.path.join(REPO_ROOT, "manual_gpt")
PROMPT_PATH = os.path.join(MANUAL_GPT_DIR, "prompts", "rewrite_v1.md")
SAMPLE_PATH = os.path.join(MANUAL_GPT_DIR, "rewrite_study_query_ids.csv")
OUTPUT_PATH = os.path.join(MANUAL_GPT_DIR, "query_rewrites.csv")


def load_prompt_template():
    with open(PROMPT_PATH) as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Automated (Groq API) query-rewrite generation for the Step 9 sample.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N queries (for a quick test)")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY environment variable not set. Get a free key at console.groq.com and set it before running.")

    prompt_template = load_prompt_template()
    sample = pd.read_csv(SAMPLE_PATH)
    if args.limit:
        sample = sample.head(args.limit)

    rows = []
    for i, row in sample.iterrows():
        prompt_text = prompt_template.replace("{QUERY}", row["raw_query"])
        raw_response = call_groq(prompt_text, api_key, args.model)
        parsed, status = parse_json_response(raw_response)

        rows.append({
            "query_id": row["query_id"],
            "raw_query": row["raw_query"],
            "prompt_version": "rewrite_v1",
            "raw_gpt_response": raw_response,
            "expanded_query_gpt": parsed.get("expanded_query", ""),
            "intent_gpt": parsed.get("intent", ""),
            "category_gpt": parsed.get("category", ""),
            "uncertainty_gpt": parsed.get("uncertainty", ""),
            "review_decision": "",
            "reviewed_query": "",
            "review_reason": "",
            "model_label": f"{args.model} (Groq API, automated generation)",
            "run_date": datetime.now(timezone.utc).isoformat(),
            "minutes_spent": "",  # automated generation - not a manual-effort measurement, see notes
            "parse_status": status,
        })
        print(f"[{i+1}/{len(sample)}] {row['raw_query']!r} -> {parsed.get('expanded_query', '?')!r} ({status})")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nsaved {OUTPUT_PATH}: {len(out_df)} rows")
    print(f"parse_status counts: {out_df['parse_status'].value_counts().to_dict()}")
    print("\nNEXT: fill in review_decision (accept/edit/reject), reviewed_query, and review_reason for each row before running the Step 9 evaluation.")


if __name__ == "__main__":
    main()
