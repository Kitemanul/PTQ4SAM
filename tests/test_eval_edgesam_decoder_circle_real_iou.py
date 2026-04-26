import argparse
import unittest

import numpy as np

from scripts.eval_edgesam_decoder_circle_real_iou import (
    build_summary_payload,
    pad_point_prompt,
)


class PadPointPromptTest(unittest.TestCase):
    def test_pads_single_point_to_static_prompt_count(self) -> None:
        coords = np.array([[12.5, 7.0]], dtype=np.float32)
        labels = np.array([1.0], dtype=np.float32)

        padded_coords, padded_labels = pad_point_prompt(coords, labels, num_points=5)

        np.testing.assert_allclose(padded_coords[0], coords[0])
        np.testing.assert_allclose(padded_coords[1:], 0.0)
        np.testing.assert_allclose(padded_labels, np.array([1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32))

    def test_rejects_too_many_points_for_static_model(self) -> None:
        coords = np.zeros((6, 2), dtype=np.float32)
        labels = np.ones((6,), dtype=np.float32)

        with self.assertRaises(ValueError):
            pad_point_prompt(coords, labels, num_points=5)


class BuildSummaryPayloadTest(unittest.TestCase):
    def test_reports_circle_and_fp32_metrics(self) -> None:
        args = argparse.Namespace(
            checkpoint='weights/edge_sam.pth',
            ann_file='annotations.json',
            img_dir='images',
            circle_model='decoder.fake_quant.circle',
            quant_circle_model=None,
            circle_interpreter_lib='libcircle_interpreter_cffi.so',
            use_stability_score=True,
            mask_threshold=None,
            fp32_mask_threshold=0.0,
            circle_mask_threshold=0.0,
            point_strategy='center',
            num_points=1,
            circle_num_points=5,
            max_images=10,
            save_per_mask=False,
        )
        per_mask = [
            {
                'fp32_selected_iou': 0.4, 'fp32_oracle_iou': 0.7, 'fp32_oracle_gap': 0.3, 'fp32_rank_hit_at_1': 0.0,
                'circle_selected_iou': 0.5, 'circle_oracle_iou': 0.8, 'circle_oracle_gap': 0.3, 'circle_rank_hit_at_1': 1.0,
            },
            {
                'fp32_selected_iou': 0.6, 'fp32_oracle_iou': 0.6, 'fp32_oracle_gap': 0.0, 'fp32_rank_hit_at_1': 1.0,
                'circle_selected_iou': 0.3, 'circle_oracle_iou': 0.9, 'circle_oracle_gap': 0.6, 'circle_rank_hit_at_1': 0.0,
            },
        ]
        per_image = [
            {
                'image_id': 1,
                'file_name': 'a.jpg',
                'num_masks': 2,
                'fp32_mean_iou': 0.5,
                'circle_mean_iou': 0.4,
                'delta_mean_iou': -0.1,
                'fp32_oracle_mean_iou': 0.65,
                'circle_oracle_mean_iou': 0.85,
                'delta_oracle_mean_iou': 0.2,
                'fp32_oracle_gap': 0.15,
                'circle_oracle_gap': 0.45,
                'fp32_rank_hit_at_1': 0.5,
                'circle_rank_hit_at_1': 0.5,
            }
        ]

        payload = build_summary_payload(args, per_mask, per_image)

        summary = payload['summary']
        self.assertAlmostEqual(summary['fp32_selected_miou'], 0.5)
        self.assertAlmostEqual(summary['circle_selected_miou'], 0.4)
        self.assertAlmostEqual(summary['delta_selected_miou'], -0.1)
        self.assertAlmostEqual(summary['circle_oracle_miou'], 0.85)
        self.assertAlmostEqual(summary['delta_oracle_miou'], 0.2)
        self.assertAlmostEqual(summary['circle_miou'], summary['circle_selected_miou'])


if __name__ == '__main__':
    unittest.main()
