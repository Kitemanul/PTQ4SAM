import json
import tempfile
import unittest
from pathlib import Path

from ptq4sam.selection_metrics import (
    aggregate_selection_metrics,
    summarize_selection_dataset,
    summarize_selection_json,
    summarize_selection_sample,
)


class SummarizeSelectionSampleTest(unittest.TestCase):
    def test_reports_selected_and_oracle_metrics(self) -> None:
        summary = summarize_selection_sample(
            candidate_ious=[0.41, 0.82, 0.67],
            candidate_scores=[0.91, 0.30, 0.22],
        )

        self.assertEqual(summary['selected_index'], 0)
        self.assertEqual(summary['oracle_index'], 1)
        self.assertAlmostEqual(summary['selected_miou'], 0.41)
        self.assertAlmostEqual(summary['oracle_miou'], 0.82)
        self.assertAlmostEqual(summary['oracle_gap'], 0.41)
        self.assertEqual(summary['rank_hit_at_1'], 0)

    def test_uses_first_index_for_ties(self) -> None:
        summary = summarize_selection_sample(
            candidate_ious=[0.60, 0.60, 0.30],
            candidate_scores=[0.50, 0.50, 0.10],
        )

        self.assertEqual(summary['selected_index'], 0)
        self.assertEqual(summary['oracle_index'], 0)
        self.assertEqual(summary['rank_hit_at_1'], 1)

    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            summarize_selection_sample(
                candidate_ious=[0.2, 0.3],
                candidate_scores=[0.1],
            )

    def test_rejects_empty_candidates(self) -> None:
        with self.assertRaises(ValueError):
            summarize_selection_sample(candidate_ious=[], candidate_scores=[])


class AggregateSelectionMetricsTest(unittest.TestCase):
    def test_aggregates_means(self) -> None:
        aggregate = aggregate_selection_metrics(
            [
                summarize_selection_sample(
                    candidate_ious=[0.41, 0.82, 0.67],
                    candidate_scores=[0.91, 0.30, 0.22],
                ),
                summarize_selection_sample(
                    candidate_ious=[0.74, 0.55, 0.31],
                    candidate_scores=[0.20, 0.80, 0.10],
                ),
            ]
        )

        self.assertEqual(aggregate['samples_evaluated'], 2)
        self.assertAlmostEqual(aggregate['selected_miou'], (0.41 + 0.55) / 2.0)
        self.assertAlmostEqual(aggregate['oracle_miou'], (0.82 + 0.74) / 2.0)
        self.assertAlmostEqual(aggregate['oracle_gap'], ((0.82 - 0.41) + (0.74 - 0.55)) / 2.0)
        self.assertAlmostEqual(aggregate['rank_hit_at_1'], 0.0)

    def test_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_selection_metrics([])


class SummarizeSelectionDatasetTest(unittest.TestCase):
    def test_returns_summary_and_per_sample_payload(self) -> None:
        payload = summarize_selection_dataset(
            [
                {'sample': 'a', 'candidate_ious': [0.1, 0.8], 'candidate_scores': [0.9, 0.1]},
                {'sample': 'b', 'candidate_ious': [0.7, 0.2], 'candidate_scores': [0.6, 0.4]},
            ]
        )

        self.assertEqual(payload['summary']['samples_evaluated'], 2)
        self.assertEqual(len(payload['per_sample']), 2)
        self.assertIn('selected_miou', payload['per_sample'][0])
        self.assertIn('oracle_miou', payload['per_sample'][0])

    def test_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'records.json'
            output_path = Path(tmpdir) / 'summary.json'
            input_path.write_text(
                json.dumps(
                    [
                        {'sample': 'a', 'candidate_ious': [0.1, 0.8], 'candidate_scores': [0.9, 0.1]},
                        {'sample': 'b', 'candidate_ious': [0.7, 0.2], 'candidate_scores': [0.6, 0.4]},
                    ]
                )
            )

            payload = summarize_selection_json(input_path, output_path)

            self.assertEqual(payload['summary']['samples_evaluated'], 2)
            written = json.loads(output_path.read_text())
            self.assertEqual(written['summary']['samples_evaluated'], 2)


if __name__ == '__main__':
    unittest.main()
