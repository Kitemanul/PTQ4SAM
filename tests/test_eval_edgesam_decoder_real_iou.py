import argparse
import unittest

import numpy as np

from scripts.eval_edgesam_decoder_real_iou import (
    aggregate_prompt_metrics,
    build_summary_payload,
    compute_iou,
    resolve_mask_thresholds,
    summarize_candidate_masks,
    summarize_image_records,
)


class ComputeIouTest(unittest.TestCase):
    def test_compute_iou(self) -> None:
        pred = np.array([[1, 0], [1, 0]], dtype=bool)
        gt = np.array([[1, 1], [0, 0]], dtype=bool)
        self.assertAlmostEqual(compute_iou(pred, gt), 1.0 / 3.0)


class SummarizeCandidateMasksTest(unittest.TestCase):
    def test_reports_selected_and_oracle_candidates(self) -> None:
        gt = np.array([[1, 1], [0, 0]], dtype=bool)
        candidates = np.array(
            [
                [[1, 0], [0, 0]],
                [[1, 1], [0, 0]],
            ],
            dtype=bool,
        )
        scores = np.array([0.9, 0.2], dtype=np.float32)

        summary = summarize_candidate_masks(candidates, scores, gt)

        self.assertEqual(summary['selected_index'], 0)
        self.assertEqual(summary['oracle_index'], 1)
        self.assertAlmostEqual(summary['selected_iou'], 0.5)
        self.assertAlmostEqual(summary['oracle_iou'], 1.0)
        self.assertAlmostEqual(summary['oracle_gap'], 0.5)
        self.assertEqual(summary['rank_hit_at_1'], 0.0)


class AggregatePromptMetricsTest(unittest.TestCase):
    def test_aggregates_summary_fields(self) -> None:
        records = [
            {'fp32_selected_iou': 0.4, 'fp32_oracle_iou': 0.7, 'fp32_oracle_gap': 0.3, 'fp32_rank_hit_at_1': 0.0},
            {'fp32_selected_iou': 0.6, 'fp32_oracle_iou': 0.6, 'fp32_oracle_gap': 0.0, 'fp32_rank_hit_at_1': 1.0},
        ]

        summary = aggregate_prompt_metrics(records, 'fp32')

        self.assertAlmostEqual(summary['fp32_selected_miou'], 0.5)
        self.assertAlmostEqual(summary['fp32_oracle_miou'], 0.65)
        self.assertAlmostEqual(summary['fp32_oracle_gap'], 0.15)
        self.assertAlmostEqual(summary['fp32_rank_hit_at_1'], 0.5)


class SummarizeImageRecordsTest(unittest.TestCase):
    def test_summarizes_selected_and_oracle_image_metrics(self) -> None:
        records = [
            {
                'fp32_selected_iou': 0.4, 'fp32_oracle_iou': 0.7, 'fp32_oracle_gap': 0.3, 'fp32_rank_hit_at_1': 0.0,
                'uint8_selected_iou': 0.2, 'uint8_oracle_iou': 0.6, 'uint8_oracle_gap': 0.4, 'uint8_rank_hit_at_1': 0.0,
            },
            {
                'fp32_selected_iou': 0.8, 'fp32_oracle_iou': 0.8, 'fp32_oracle_gap': 0.0, 'fp32_rank_hit_at_1': 1.0,
                'uint8_selected_iou': 0.5, 'uint8_oracle_iou': 0.9, 'uint8_oracle_gap': 0.4, 'uint8_rank_hit_at_1': 1.0,
            },
        ]

        summary = summarize_image_records(1, 'a.jpg', records)

        self.assertEqual(summary['num_masks'], 2)
        self.assertAlmostEqual(summary['fp32_mean_iou'], 0.6)
        self.assertAlmostEqual(summary['uint8_mean_iou'], 0.35)
        self.assertAlmostEqual(summary['fp32_oracle_mean_iou'], 0.75)
        self.assertAlmostEqual(summary['uint8_oracle_mean_iou'], 0.75)
        self.assertAlmostEqual(summary['delta_mean_iou'], -0.25)
        self.assertAlmostEqual(summary['delta_oracle_mean_iou'], 0.0)


class ResolveMaskThresholdsTest(unittest.TestCase):
    def test_prefers_model_specific_thresholds_over_shared_threshold(self) -> None:
        args = argparse.Namespace(mask_threshold=0.1, fp32_mask_threshold=-0.05, uint8_mask_threshold=0.2)

        fp32_threshold, uint8_threshold = resolve_mask_thresholds(args, 0.0)

        self.assertAlmostEqual(fp32_threshold, -0.05)
        self.assertAlmostEqual(uint8_threshold, 0.2)

    def test_falls_back_to_default_threshold(self) -> None:
        args = argparse.Namespace(mask_threshold=None, fp32_mask_threshold=None, uint8_mask_threshold=None)

        fp32_threshold, uint8_threshold = resolve_mask_thresholds(args, 0.0)

        self.assertAlmostEqual(fp32_threshold, 0.0)
        self.assertAlmostEqual(uint8_threshold, 0.0)


class BuildSummaryPayloadTest(unittest.TestCase):
    def test_builds_backward_compatible_selected_aliases(self) -> None:
        args = argparse.Namespace(
            checkpoint='weights/edge_sam.pth',
            ann_file='annotations.json',
            img_dir='images',
            calibration_list='calib.txt',
            bit=8,
            scope='full',
            disable_big=False,
            disable_agq=True,
            use_stability_score=True,
            quantize_hypernetwork_output=True,
            quantize_mask_projection_hyper_input=True,
            quantize_mask_projection_upscaled_input=True,
            hypernetwork_output_bit=None,
            mask_projection_hyper_input_bit=None,
            mask_projection_upscaled_input_bit=None,
            mask_threshold=None,
            fp32_mask_threshold=None,
            uint8_mask_threshold=None,
            point_strategy='center',
            num_points=1,
            max_images=10,
            save_per_mask=False,
        )
        per_mask = [
            {
                'fp32_selected_iou': 0.4, 'fp32_oracle_iou': 0.7, 'fp32_oracle_gap': 0.3, 'fp32_rank_hit_at_1': 0.0,
                'uint8_selected_iou': 0.2, 'uint8_oracle_iou': 0.6, 'uint8_oracle_gap': 0.4, 'uint8_rank_hit_at_1': 0.0,
            },
            {
                'fp32_selected_iou': 0.8, 'fp32_oracle_iou': 0.8, 'fp32_oracle_gap': 0.0, 'fp32_rank_hit_at_1': 1.0,
                'uint8_selected_iou': 0.5, 'uint8_oracle_iou': 0.9, 'uint8_oracle_gap': 0.4, 'uint8_rank_hit_at_1': 1.0,
            },
        ]
        per_image = [
            {'image_id': 1, 'file_name': 'a.jpg', 'num_masks': 2, 'fp32_mean_iou': 0.6, 'uint8_mean_iou': 0.35, 'delta_mean_iou': -0.25, 'fp32_oracle_mean_iou': 0.75, 'uint8_oracle_mean_iou': 0.75, 'delta_oracle_mean_iou': 0.0, 'fp32_oracle_gap': 0.15, 'uint8_oracle_gap': 0.4, 'fp32_rank_hit_at_1': 0.5, 'uint8_rank_hit_at_1': 0.5},
        ]

        payload = build_summary_payload(args, per_mask, per_image)

        summary = payload['summary']
        self.assertAlmostEqual(summary['fp32_selected_miou'], 0.6)
        self.assertAlmostEqual(summary['uint8_selected_miou'], 0.35)
        self.assertAlmostEqual(summary['delta_selected_miou'], -0.25)
        self.assertAlmostEqual(summary['fp32_oracle_miou'], 0.75)
        self.assertAlmostEqual(summary['uint8_oracle_miou'], 0.75)
        self.assertAlmostEqual(summary['delta_oracle_miou'], 0.0)
        self.assertAlmostEqual(summary['fp32_miou'], summary['fp32_selected_miou'])
        self.assertAlmostEqual(summary['uint8_miou'], summary['uint8_selected_miou'])
        self.assertAlmostEqual(summary['delta_miou'], summary['delta_selected_miou'])


if __name__ == '__main__':
    unittest.main()
