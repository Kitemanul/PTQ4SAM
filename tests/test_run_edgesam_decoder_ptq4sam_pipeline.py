import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "wandb" not in sys.modules:
    sys.modules["wandb"] = types.ModuleType("wandb")

from scripts.run_edgesam_decoder_ptq4sam_pipeline import ensure_wandb_stub, main  # noqa: E402


class _DummySurface(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eval_called = False

    def eval(self) -> "_DummySurface":
        self.eval_called = True
        return self

    def to(self, device) -> "_DummySurface":
        return self


class RunPipelineMainTest(unittest.TestCase):
    def test_ensure_wandb_stub_inserts_module(self) -> None:
        original = sys.modules.pop("wandb", None)
        try:
            ensure_wandb_stub()
            self.assertIn("wandb", sys.modules)
            self.assertIsInstance(sys.modules["wandb"], types.ModuleType)
        finally:
            if original is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = original

    def test_main_passes_dummy_inputs_to_export_and_adapts_dense_embedding(self) -> None:
        fp_model = _DummySurface()
        quant_model = _DummySurface()
        dummy_inputs = (
            torch.randn(1, 256, 32, 32),
            torch.randn(1, 5, 256),
            torch.randint(0, 4, (1, 5), dtype=torch.int64).to(torch.float32),
        )
        bundle = SimpleNamespace(mean_metrics={"scores_mae": 0.1}, per_sample=[{"sample": "a"}])

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                checkpoint="weights/edge_sam.pth",
                calibration_list="weights/calib.txt",
                eval_list="weights/eval.txt",
                calibration_count=1,
                eval_count=1,
                bit=8,
                scope="transformer",
                disable_big=False,
                disable_agq=False,
                summary_output_dir=tmpdir,
                onnx_output=str(Path(tmpdir) / "decoder.onnx"),
                image_embeddings_shape=(1, 256, 32, 32),
                point_embedding_pe_shape=(1, 5, 256),
                point_labels_shape=(1, 5),
                opset_version=11,
                check_ops_only=False,
            )

            with patch("scripts.run_edgesam_decoder_ptq4sam_pipeline.parse_args", return_value=args), patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline.torch.cuda.is_available",
                return_value=True,
            ), patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline.collect_decoder_sample_triplets",
                side_effect=[[("a", "b", "c")], [("d", "e", "f")]],
            ), patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline._build_decoder_surface",
                side_effect=[fp_model, quant_model],
            ), patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline.quantize_decoder_surface",
                return_value=quant_model,
            ), patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline.move_samples_to_device",
                return_value=[(torch.ones(1), torch.ones(1), torch.ones(1))],
            ), patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline.calibrate_decoder"
            ), patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline.enable_quantization"
            ), patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline.evaluate_decoder",
                return_value=bundle,
            ), patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline.resolve_output_path",
                return_value=Path(tmpdir) / "decoder.onnx",
            ), patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline.build_dummy_inputs",
                return_value=dummy_inputs,
            ) as build_dummy_mock, patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline.adapt_decoder_dense_embedding_size"
            ) as adapt_mock, patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline.export_quantized_decoder_to_onnx"
            ) as export_mock, patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline.export_pe_gaussian_matrix",
                return_value=Path(tmpdir) / "pe_gaussian_matrix.bin",
            ), patch(
                "scripts.run_edgesam_decoder_ptq4sam_pipeline._print_onnx_summary"
            ):
                main()

        build_dummy_mock.assert_called_once_with(args)
        adapt_mock.assert_called_once_with(quant_model, image_h=32, image_w=32)
        _, kwargs = export_mock.call_args
        self.assertIs(kwargs["dummy_inputs"], dummy_inputs)
        self.assertEqual(kwargs["num_points"], 5)
        self.assertEqual(kwargs["opset_version"], 11)


if __name__ == "__main__":
    unittest.main()
