from __future__ import annotations

import argparse
import os
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_wandb_stub() -> None:
    if "wandb" not in sys.modules:
        sys.modules["wandb"] = types.ModuleType("wandb")


ensure_wandb_stub()

from edge_sam import sam_model_registry
from edge_sam.quantization.decoder_ptq_compare import _build_decoder_surface, collect_decoder_sample_triplets
from ptq4sam.quantization.fake_quant import AdaptiveGranularityQuantize, QuantizeBase
from ptq4sam.quantization.quantized_module import PreQuantizedLayer, QuantizedMatMul
from scripts.edgesam_decoder_ptq4sam_uint8 import (
    calibrate_decoder,
    make_qconfig,
    move_samples_to_device,
    quantize_decoder_surface,
)
from ptq4sam.quantization.state import enable_quantization


def resolve_output_path(checkpoint: str, output: str | None, scope: str) -> Path:
    if output is not None:
        return Path(output)
    checkpoint_path = Path(checkpoint)
    stem = checkpoint_path.stem
    return checkpoint_path.with_name(f"{stem}_decoder_ptq4sam_fp32_{scope}.onnx")


def export_pe_gaussian_matrix(checkpoint: str, output_path: Path) -> Path:
    sam = sam_model_registry["edge_sam"](checkpoint=checkpoint, upsample_mode="bilinear")
    sam.eval()
    pe_matrix = sam.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix.cpu().numpy()
    pe_path = output_path.parent / "pe_gaussian_matrix.bin"
    pe_matrix.astype(np.float32).tofile(pe_path)
    return pe_path


def _onnx_export(*args, **kwargs) -> None:
    parts = torch.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    if major > 2 or (major == 2 and minor >= 6):
        kwargs["dynamo"] = False
    kwargs.setdefault("do_constant_folding", False)
    torch.onnx.export(*args, **kwargs)


def _export_dtype(quant_min: int, quant_max: int) -> torch.dtype:
    return torch.uint8 if quant_min >= 0 and quant_max <= 255 else torch.int8


def _reshape_qparam(param: torch.Tensor, x: torch.Tensor, ch_axis: int) -> torch.Tensor:
    if param.numel() == 1:
        if x.dim() == 0:
            return param.reshape(())
        return param.reshape([1] * x.dim())
    if ch_axis == -1:
        return param
    shape = [1] * x.dim()
    shape[ch_axis] = x.shape[ch_axis]
    return param.reshape(shape)


def _clone_zero_point(module: QuantizeBase) -> torch.Tensor:
    zero_point = getattr(module, "zero_point", None)
    if zero_point is None:
        scale = module.scale.detach().clone()
        shape = scale.shape if scale.ndim > 0 else (1,)
        return torch.zeros(shape, dtype=torch.int32)
    return zero_point.detach().clone().to(torch.int32)


def _round_nearest_without_round(x: torch.Tensor) -> torch.Tensor:
    positive = torch.floor(x + 0.5)
    negative = -torch.floor((-x) + 0.5)
    return torch.where(x < 0, negative, positive)


def _quantize_tensor_to_dtype(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    quant_min: int,
    quant_max: int,
    ch_axis: int,
    qdtype: torch.dtype,
) -> torch.Tensor:
    scale_view = _reshape_qparam(scale.to(x.dtype), x, ch_axis)
    zero_view = _reshape_qparam(zero_point.to(x.dtype), x, ch_axis)
    x_q = torch.clamp(torch.round(x / scale_view) + zero_view, quant_min, quant_max)
    return x_q.to(qdtype)


def _dequantize_tensor_from_dtype(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    ch_axis: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    scale_view = _reshape_qparam(scale.to(dtype), x_q, ch_axis)
    zero_view = _reshape_qparam(zero_point.to(dtype), x_q, ch_axis)
    return (x_q.to(dtype) - zero_view) * scale_view


class ExportAffineQuantizer(nn.Module):
    def __init__(self, module: QuantizeBase) -> None:
        super().__init__()
        self.quant_min = module.quant_min
        self.quant_max = module.quant_max
        self.ch_axis = module.ch_axis
        self.register_buffer("scale", module.scale.detach().clone().to(torch.float32))
        self.register_buffer("zero_point", _clone_zero_point(module))

    def forward(self, x: torch.Tensor, value=None) -> torch.Tensor:
        del value
        scale_view = _reshape_qparam(self.scale.to(x.dtype), x, self.ch_axis)
        zero_view = _reshape_qparam(self.zero_point.to(x.dtype), x, self.ch_axis)
        x_q = torch.clamp(_round_nearest_without_round(x / scale_view) + zero_view, self.quant_min, self.quant_max)
        return (x_q - zero_view) * scale_view


class ExportAdaptiveGranularityQuantizer(nn.Module):
    def __init__(self, module: AdaptiveGranularityQuantize) -> None:
        super().__init__()
        self.quant_min = module.quant_min
        self.quant_max = module.quant_max
        self.tau = float(module.tau)
        self.register_buffer("scale", module.scale.detach().clone().to(torch.float32))
        self.register_buffer("zero_point", _clone_zero_point(module))

    def forward(self, x: torch.Tensor, value=None) -> torch.Tensor:
        del value
        levels = self.quant_max - self.quant_min + 1
        scale = self.scale.to(x.dtype)
        x_safe = torch.clamp(x, 1e-20, None)
        x_int = _round_nearest_without_round(-torch.log2(x_safe / scale) * self.tau)
        softmax_mask = x_int >= levels
        x_q = torch.clamp(x_int, 0, levels - 1)
        dequant = scale * torch.pow(x.new_tensor(2.0), -x_q / self.tau)
        return torch.where(softmax_mask, torch.zeros_like(dequant), dequant)


class ExportWeightBackedLinear(nn.Module):
    def __init__(self, module: nn.Linear) -> None:
        super().__init__()
        quantizer = module.weight_fake_quant
        self.quant_min = quantizer.quant_min
        self.quant_max = quantizer.quant_max
        self.ch_axis = quantizer.ch_axis
        self.qdtype = _export_dtype(self.quant_min, self.quant_max)
        self.register_buffer("scale", quantizer.scale.detach().clone().to(torch.float32))
        self.register_buffer("zero_point", _clone_zero_point(quantizer))
        self.register_buffer(
            "weight_q",
            _quantize_tensor_to_dtype(
                module.weight.detach().clone().to(torch.float32),
                self.scale,
                self.zero_point,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                ch_axis=self.ch_axis,
                qdtype=self.qdtype,
            ),
        )
        if module.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", module.bias.detach().clone().to(torch.float32))

    def forward(self, input: torch.Tensor, gamma=None) -> torch.Tensor:
        weight = _dequantize_tensor_from_dtype(
            self.weight_q,
            self.scale,
            self.zero_point,
            ch_axis=self.ch_axis,
            dtype=input.dtype,
        )
        bias = self.bias
        if gamma is not None:
            weight = weight * gamma.unsqueeze(1)
            if bias is not None:
                bias = bias.to(input.dtype) * gamma.to(input.dtype)
        elif bias is not None:
            bias = bias.to(input.dtype)
        return F.linear(input, weight, bias)


class ExportWeightBackedConvTranspose2d(nn.Module):
    def __init__(self, module: nn.ConvTranspose2d) -> None:
        super().__init__()
        quantizer = module.weight_fake_quant
        self.quant_min = quantizer.quant_min
        self.quant_max = quantizer.quant_max
        self.ch_axis = quantizer.ch_axis
        self.qdtype = _export_dtype(self.quant_min, self.quant_max)
        self.register_buffer("scale", quantizer.scale.detach().clone().to(torch.float32))
        self.register_buffer("zero_point", _clone_zero_point(quantizer))
        self.register_buffer(
            "weight_q",
            _quantize_tensor_to_dtype(
                module.weight.detach().clone().to(torch.float32),
                self.scale,
                self.zero_point,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                ch_axis=self.ch_axis,
                qdtype=self.qdtype,
            ),
        )
        if module.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", module.bias.detach().clone().to(torch.float32))
        self.stride = module.stride
        self.padding = module.padding
        self.output_padding = module.output_padding
        self.groups = module.groups
        self.dilation = module.dilation

    def forward(self, input: torch.Tensor, gamma=None) -> torch.Tensor:
        del gamma
        weight = _dequantize_tensor_from_dtype(
            self.weight_q,
            self.scale,
            self.zero_point,
            ch_axis=self.ch_axis,
            dtype=input.dtype,
        )
        bias = None if self.bias is None else self.bias.to(input.dtype)
        return F.conv_transpose2d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


class ExportWeightBackedEmbedding(nn.Module):
    def __init__(self, module: nn.Embedding) -> None:
        super().__init__()
        quantizer = module.weight_fake_quant
        self.quant_min = quantizer.quant_min
        self.quant_max = quantizer.quant_max
        self.ch_axis = quantizer.ch_axis
        self.qdtype = _export_dtype(self.quant_min, self.quant_max)
        self.register_buffer("scale", quantizer.scale.detach().clone().to(torch.float32))
        self.register_buffer("zero_point", _clone_zero_point(quantizer))
        self.register_buffer(
            "weight_q",
            _quantize_tensor_to_dtype(
                module.weight.detach().clone().to(torch.float32),
                self.scale,
                self.zero_point,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                ch_axis=self.ch_axis,
                qdtype=self.qdtype,
            ),
        )
        self.padding_idx = module.padding_idx
        self.max_norm = module.max_norm
        self.norm_type = module.norm_type
        self.scale_grad_by_freq = module.scale_grad_by_freq
        self.sparse = module.sparse

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = _dequantize_tensor_from_dtype(
            self.weight_q,
            self.scale,
            self.zero_point,
            ch_axis=self.ch_axis,
            dtype=torch.float32,
        )
        return F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class ExportPreQuantizedLayer(nn.Module):
    def __init__(self, module: PreQuantizedLayer) -> None:
        super().__init__()
        self.qinput = module.qinput
        if self.qinput:
            self.layer_pre_act_fake_quantize = _convert_module_for_uint8_export(module.layer_pre_act_fake_quantize)
        self.module = _convert_module_for_uint8_export(module.module)
        self.activation = module.activation

    def forward(self, x: torch.Tensor, gamma=None) -> torch.Tensor:
        if self.qinput:
            x = self.layer_pre_act_fake_quantize(x)
        x = self.module(x, gamma)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ExportQuantizedMatMul(nn.Module):
    def __init__(self, module: QuantizedMatMul) -> None:
        super().__init__()
        self.qinput = module.qinput
        if self.qinput:
            self.a_layer_pre_act_fake_quantize = _convert_module_for_uint8_export(module.a_layer_pre_act_fake_quantize)
            self.b_layer_pre_act_fake_quantize = _convert_module_for_uint8_export(module.b_layer_pre_act_fake_quantize)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        a, b = inputs
        if self.qinput:
            a = self.a_layer_pre_act_fake_quantize(a)
            b = self.b_layer_pre_act_fake_quantize(b)
        return a @ b


def _convert_module_for_uint8_export(module: nn.Module) -> nn.Module:
    if isinstance(module, PreQuantizedLayer):
        return ExportPreQuantizedLayer(module)
    if isinstance(module, QuantizedMatMul):
        return ExportQuantizedMatMul(module)
    if isinstance(module, AdaptiveGranularityQuantize):
        if not hasattr(module, "zero_point"):
            module.register_buffer("zero_point", _clone_zero_point(module))
        return ExportAdaptiveGranularityQuantizer(module)
    if isinstance(module, QuantizeBase):
        return ExportAffineQuantizer(module)
    if isinstance(module, nn.Linear) and hasattr(module, "weight_fake_quant"):
        return ExportWeightBackedLinear(module)
    if isinstance(module, nn.ConvTranspose2d) and hasattr(module, "weight_fake_quant"):
        return ExportWeightBackedConvTranspose2d(module)
    if isinstance(module, nn.Embedding) and hasattr(module, "weight_fake_quant"):
        return ExportWeightBackedEmbedding(module)
    for name, child in list(module.named_children()):
        setattr(module, name, _convert_module_for_uint8_export(child))
    return module


def prepare_quantized_decoder_for_onnx_export(decoder: torch.nn.Module) -> torch.nn.Module:
    def convert(module: nn.Module) -> nn.Module:
        if isinstance(module, AdaptiveGranularityQuantize):
            if not hasattr(module, "zero_point"):
                module.register_buffer("zero_point", _clone_zero_point(module))
            return ExportAdaptiveGranularityQuantizer(module)
        if isinstance(module, QuantizeBase):
            if not hasattr(module, "zero_point"):
                module.register_buffer("zero_point", _clone_zero_point(module))
            return ExportAffineQuantizer(module)
        for name, child in list(module.named_children()):
            setattr(module, name, convert(child))
        return module

    return convert(decoder).eval()


def _print_onnx_summary(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("Cannot print summary: onnx not installed")
        return

    dtype_names = {
        1: "FLOAT",
        2: "UINT8",
        3: "INT8",
        5: "INT16",
        6: "INT32",
        7: "INT64",
        9: "BOOL",
        10: "FLOAT16",
    }
    model = onnx.load(str(onnx_path))
    ops = Counter(node.op_type for node in model.graph.node)

    dtypes = set()
    for init in model.graph.initializer:
        dtypes.add(dtype_names.get(init.data_type, "?"))
    for inp in model.graph.input:
        dtypes.add(dtype_names.get(inp.type.tensor_type.elem_type, "?"))
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "value" and hasattr(attr, "t") and attr.t.data_type:
                dtypes.add(dtype_names.get(attr.t.data_type, "?"))

    print(f"\n{'=' * 60}")
    print(f"  {onnx_path.name}")
    print(f"{'=' * 60}")
    print(f"  Nodes: {len(model.graph.node)},  Op types: {len(ops)}")
    print(f"  Ops: {', '.join(f'{op}({cnt})' for op, cnt in sorted(ops.items()))}")
    print(f"  Dtypes: {', '.join(sorted(dtypes))}")
    for inp in model.graph.input:
        shape = [dim.dim_value or dim.dim_param for dim in inp.type.tensor_type.shape.dim]
        dt = dtype_names.get(inp.type.tensor_type.elem_type, "?")
        print(f"  Input  {inp.name}: {dt} {shape}")
    for out in model.graph.output:
        shape = [dim.dim_value or dim.dim_param for dim in out.type.tensor_type.shape.dim]
        dt = dtype_names.get(out.type.tensor_type.elem_type, "?")
        print(f"  Output {out.name}: {dt} {shape}")

    known_npu_bad = {"Sin", "Cos", "Abs", "Gather", "GatherND", "ScatterND", "NonZero", "Greater"}
    likely_quant_bad = {"Round", "Clip", "Log"}
    found = (known_npu_bad | likely_quant_bad) & set(ops.keys())
    if found:
        print(f"  WARNING: ops needing review for NPU compilation: {', '.join(sorted(found))}")
    if "INT64" in dtypes:
        print("  WARNING: INT64 data type present")
    print(f"{'=' * 60}")


def build_quantized_decoder_for_export(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    checkpoint_path = Path(args.checkpoint).resolve()
    calibration_paths = collect_decoder_sample_triplets(args.calibration_list, limit=args.calibration_count)
    if len(calibration_paths) < args.calibration_count:
        raise ValueError(
            f"Requested {args.calibration_count} calibration samples but found {len(calibration_paths)}"
        )

    config_quant = make_qconfig(bit=args.bit)
    config_quant.ptq4sam.BIG = not args.disable_big
    config_quant.ptq4sam.AGQ = not args.disable_agq

    decoder = _build_decoder_surface(checkpoint_path, use_stability_score=args.use_stability_score).to(device).eval()
    decoder = quantize_decoder_surface(decoder, config_quant, scope=args.scope).to(device).eval()

    calibration_samples = move_samples_to_device(calibration_paths, device)
    calibrate_decoder(decoder, calibration_samples, config_quant.ptq4sam.BIG)
    enable_quantization(decoder)
    return decoder


def export_quantized_decoder_to_onnx(
    decoder: torch.nn.Module,
    output_path: Path,
    *,
    num_points: int,
    opset_version: int,
    dummy_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    decoder = prepare_quantized_decoder_for_onnx_export(decoder)
    if dummy_inputs is None:
        embed_dim = getattr(decoder, "dense_embedding").shape[1]
        embed_h = int(getattr(decoder, "embed_h", getattr(decoder, "dense_embedding").shape[2]))
        embed_w = int(getattr(decoder, "embed_w", getattr(decoder, "dense_embedding").shape[3]))
        dummy_embeddings = torch.randn(1, embed_dim, embed_h, embed_w, dtype=torch.float32)
        dummy_pe = torch.randn(1, num_points, embed_dim, dtype=torch.float32)
        dummy_labels = torch.randint(0, 4, (1, num_points), dtype=torch.int64).to(torch.float32)
    else:
        dummy_embeddings, dummy_pe, dummy_labels = dummy_inputs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        _onnx_export(
            decoder.cpu().eval(),
            (dummy_embeddings, dummy_pe, dummy_labels),
            str(output_path),
            input_names=["image_embeddings", "point_embedding_pe", "point_labels"],
            output_names=["scores", "masks"],
            opset_version=opset_version,
            verbose=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a PTQ4SAM-style EdgeSAM decoder ONNX as FP32, preserving BIG/AGQ fake-quant math."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="/home/kitemanul/project/EdgeSAM/weights/edge_sam.pth",
        help="Path to EdgeSAM checkpoint (.pth)",
    )
    parser.add_argument(
        "--calibration-list",
        default="/home/kitemanul/project/EdgeSAM/weights/one_pipeline/calib/decoder_datalist_20.txt",
        help="Decoder calibration datalist",
    )
    parser.add_argument("--calibration-count", type=int, default=20)
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument(
        "--scope",
        choices=("transformer", "full"),
        default="transformer",
        help="Quantization boundary to export",
    )
    parser.add_argument("--disable-big", action="store_true")
    parser.add_argument("--disable-agq", action="store_true")
    parser.add_argument("--use-stability-score", action="store_true", default=True)
    parser.add_argument("--num-points", type=int, default=5, help="Static prompt count for exported ONNX")
    parser.add_argument("--opset-version", type=int, default=11)
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path")
    parser.add_argument(
        "--check-ops-only",
        action="store_true",
        help="Export, print ONNX summary, then delete the file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because PTQ4SAM BIG/AGQ calibration code uses CUDA tensors")

    device = torch.device("cuda:0")
    output_path = resolve_output_path(args.checkpoint, args.output, args.scope)
    decoder = build_quantized_decoder_for_export(args, device=device)
    export_quantized_decoder_to_onnx(
        decoder,
        output_path,
        num_points=args.num_points,
        opset_version=args.opset_version,
    )
    pe_path = export_pe_gaussian_matrix(args.checkpoint, output_path)
    print(f"Exported quantized decoder ONNX to {output_path}")
    print(f"Exported PE gaussian matrix to {pe_path}")
    print("Note: this is a PTQ4SAM FP32 ONNX traced from the calibrated fake-quant model, not a uint8-backed graph.")
    _print_onnx_summary(output_path)
    if args.check_ops_only:
        os.remove(output_path)
        print(f"\n(Removed {output_path} — check-ops-only mode)")


if __name__ == "__main__":
    main()
