import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from ptq4sam.quantization.fake_quant import AdaptiveGranularityQuantize
from ptq4sam.quantization.observer import ObserverBase
from ptq4sam.quantization.quantized_module import PreQuantizedLayer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "wandb" not in sys.modules:
    sys.modules["wandb"] = types.ModuleType("wandb")

from scripts.export_edgesam_decoder_ptq4sam_onnx import (  # noqa: E402
    _onnx_export,
    build_quantized_decoder_for_export,
    prepare_quantized_decoder_for_onnx_export,
    resolve_output_path,
)
from scripts.edgesam_decoder_ptq4sam_uint8 import make_qconfig

try:
    import onnx
except ImportError:  # pragma: no cover
    onnx = None


class _DummySurface(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eval_called = False

    def eval(self) -> "_DummySurface":
        self.eval_called = True
        return self

    def to(self, device) -> "_DummySurface":
        return self


class ResolveOutputPathTest(unittest.TestCase):
    def test_uses_fp32_decoder_suffix_when_output_not_provided(self) -> None:
        result = resolve_output_path("weights/edge_sam.pth", None, scope="transformer")

        self.assertEqual(result, Path("weights/edge_sam_decoder_ptq4sam_fp32_transformer.onnx"))

    def test_preserves_explicit_output_path(self) -> None:
        result = resolve_output_path("weights/edge_sam.pth", "tmp/custom.onnx", scope="full")

        self.assertEqual(result, Path("tmp/custom.onnx"))


class OnnxExportOptionsTest(unittest.TestCase):
    def test_disables_constant_folding_to_preserve_fake_quant_float_math(self) -> None:
        with patch("scripts.export_edgesam_decoder_ptq4sam_onnx.torch.onnx.export") as export_mock:
            _onnx_export("model", "args", output="toy.onnx")

        _, kwargs = export_mock.call_args
        self.assertFalse(kwargs["do_constant_folding"])


class BuildQuantizedDecoderForExportTest(unittest.TestCase):
    def test_builds_calibrated_quantized_surface_with_big_and_agq(self) -> None:
        surface = _DummySurface()
        quantized_surface = _DummySurface()
        calibration_samples = [(torch.ones(1), torch.ones(1), torch.ones(1))]
        args = SimpleNamespace(
            checkpoint="weights/edge_sam.pth",
            calibration_list="weights/calib.txt",
            calibration_count=3,
            bit=8,
            scope="transformer",
            disable_big=False,
            disable_agq=False,
            use_stability_score=True,
        )

        with patch(
            "scripts.export_edgesam_decoder_ptq4sam_onnx._build_decoder_surface",
            return_value=surface,
        ), patch(
            "scripts.export_edgesam_decoder_ptq4sam_onnx.collect_decoder_sample_triplets",
            return_value=[("a", "b", "c")] * 3,
        ), patch(
            "scripts.export_edgesam_decoder_ptq4sam_onnx.move_samples_to_device",
            return_value=calibration_samples,
        ), patch(
            "scripts.export_edgesam_decoder_ptq4sam_onnx.quantize_decoder_surface",
            return_value=quantized_surface,
        ) as quantize_mock, patch(
            "scripts.export_edgesam_decoder_ptq4sam_onnx.calibrate_decoder"
        ) as calibrate_mock, patch(
            "scripts.export_edgesam_decoder_ptq4sam_onnx.enable_quantization"
        ) as enable_quant_mock:
            result = build_quantized_decoder_for_export(args, device=torch.device("cpu"))

        self.assertIs(result, quantized_surface)
        quantize_args, quantize_kwargs = quantize_mock.call_args
        self.assertIs(quantize_args[0], surface)
        self.assertTrue(quantize_args[1].ptq4sam.BIG)
        self.assertTrue(quantize_args[1].ptq4sam.AGQ)
        self.assertEqual(quantize_kwargs["scope"], "transformer")
        calibrate_mock.assert_called_once_with(quantized_surface, calibration_samples, True)
        enable_quant_mock.assert_called_once_with(quantized_surface)


class PrepareQuantizedDecoderForOnnxExportTest(unittest.TestCase):
    def test_registers_zero_point_for_agq_modules_without_uint8_rewrite(self) -> None:
        module = AdaptiveGranularityQuantize(ObserverBase, bit=8, symmetric=False, ch_axis=-1)
        wrapper = nn.Sequential(module)

        self.assertFalse(hasattr(module, "zero_point"))

        prepared = prepare_quantized_decoder_for_onnx_export(wrapper)

        self.assertIs(prepared, wrapper)
        self.assertTrue(hasattr(module, "zero_point"))
        self.assertTrue(torch.equal(module.zero_point, torch.tensor([0], dtype=torch.int32)))

    def test_keeps_prequantized_linear_without_uint8_weight_buffers(self) -> None:
        qconfig = make_qconfig()

        class _ToyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = PreQuantizedLayer(nn.Linear(4, 3), None, qconfig.w_qconfig, qconfig.a_qconfig)

        wrapper = _ToyModule()

        prepared = prepare_quantized_decoder_for_onnx_export(wrapper)

        self.assertIs(prepared, wrapper)
        self.assertFalse(hasattr(prepared.layer.module, "weight_q"))

    @unittest.skipIf(onnx is None, "onnx is required for export inspection")
    def test_exports_quantized_toy_model_as_fp32_onnx_without_uint8_initializers(self) -> None:
        qconfig = make_qconfig()

        class _ToyQuantizedModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = PreQuantizedLayer(nn.Linear(4, 4), None, qconfig.w_qconfig, qconfig.a_qconfig)
                self.softmax_post_act_fake_quantize = AdaptiveGranularityQuantize(
                    ObserverBase,
                    bit=8,
                    symmetric=False,
                    ch_axis=-1,
                )
                self._freeze_affine_quantizer(self.layer.layer_pre_act_fake_quantize, 0.125, 0)
                self._freeze_affine_quantizer(self.layer.module.weight_fake_quant, 0.0625, 0, numel=4)
                self._freeze_agq(self.softmax_post_act_fake_quantize, 0.5, 2.0)

            @staticmethod
            def _freeze_affine_quantizer(module: nn.Module, scale: float, zero_point: int, numel: int = 1) -> None:
                module.disable_observer()
                module.enable_fake_quant()
                scale_tensor = torch.full((numel,), scale, dtype=torch.float32)
                if isinstance(module.scale, torch.nn.Parameter):
                    module.scale = torch.nn.Parameter(scale_tensor)
                else:
                    module.scale.resize_((numel,))
                    module.scale.copy_(scale_tensor)
                module.zero_point.resize_((numel,))
                module.zero_point.copy_(torch.full((numel,), zero_point, dtype=torch.int32))

            @staticmethod
            def _freeze_agq(module: AdaptiveGranularityQuantize, scale: float, tau: float) -> None:
                module.disable_observer()
                module.enable_fake_quant()
                module.inited = True
                module.tau = tau
                module.scale.data.resize_((1,))
                module.scale.data.copy_(torch.tensor([scale], dtype=torch.float32))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.layer(x)
                x = torch.softmax(x, dim=-1)
                return self.softmax_post_act_fake_quantize(x)

        model = prepare_quantized_decoder_for_onnx_export(_ToyQuantizedModule()).eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "toy_fp32.onnx"
            with torch.no_grad():
                _onnx_export(
                    model,
                    (torch.randn(2, 4),),
                    str(output_path),
                    input_names=["x"],
                    output_names=["y"],
                    opset_version=11,
                    verbose=False,
                )

            graph = onnx.load(str(output_path)).graph
            init_dtypes = {initializer.data_type for initializer in graph.initializer}
            op_types = {node.op_type for node in graph.node}
            self.assertNotIn(2, init_dtypes)
            self.assertNotIn(3, init_dtypes)
            self.assertEqual(graph.input[0].type.tensor_type.elem_type, 1)
            self.assertEqual(graph.output[0].type.tensor_type.elem_type, 1)
            self.assertNotIn("Round", op_types)


if __name__ == "__main__":
    unittest.main()
