from __future__ import annotations

import argparse
import json

from ptq4sam.selection_metrics import summarize_selection_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize selected-vs-oracle mask metrics from per-sample candidate IoUs and scores."
    )
    parser.add_argument('input_json', help="JSON list with candidate_ious and candidate_scores per sample")
    parser.add_argument('--output-json', default=None, help="Optional output path for full summary payload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize_selection_json(args.input_json, args.output_json)
    print(json.dumps(payload['summary'], indent=2))


if __name__ == '__main__':
    main()
