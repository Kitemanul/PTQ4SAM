from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import types
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptq4sam.quantization.quantized_module import (
    PreQuantizedLayer,
    QuantizedBlock,
    QuantizedMatMul,
    Quantizer,
    WeightQuantizer,
)
from ptq4sam.quantization.state import enable_calibration_woquantization, enable_quantization

if "wandb" not in sys.modules:
    sys.modules["wandb"] = types.ModuleType("wandb")

from edge_sam.modeling.transformer import Attention, TwoWayAttentionBlock, TwoWayTransformer
from edge_sam.quantization.decoder_ptq_compare import (
    DecoderHypernetworkStack,
    DecoderLabelEmbedding,
    DecoderScoreHead,
    _build_decoder_surface,
    _run_decoder_model,
    collect_decoder_sample_triplets,
    compute_decoder_metrics,
    load_decoder_sample_triplet,
)


LOGGER = logging.getLogger("edgesam_decoder_ptq4sam")


class AttrDict(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def make_qconfig(bit: int = 8) -> AttrDict:
    return AttrDict(
        w_qconfig=AttrDict(
            quantizer="AdaRoundFakeQuantize",
            observer="MSEObserver",
            bit=bit,
            symmetric=False,
            ch_axis=0,
        ),
        a_qconfig=AttrDict(
            quantizer="LSQFakeQuantize",
            observer="AvgMinMaxObserver",
            bit=bit,
            symmetric=False,
            ch_axis=-1,
        ),
        ptq4sam=AttrDict(
            BIG=True,
            AGQ=True,
            global_num=128,
            peak_distance=32,
            peak_height=0.01,
        ),
    )


def update_specialized_quantizer_config(base_config: AttrDict, quantizer_name: str) -> AttrDict:
    specialized_config = AttrDict(copy.deepcopy(base_config))
    update_keys = {
        "softmax": {
            "quantizer": "AdaptiveGranularityQuantize",
            "observer": "LogAvgMSEFastObserver",
        },
        "bimodal": {
            "quantizer": "LSQSignFakeQuantize",
            "observer": "SignAvgMSEFastObserver",
        },
    }[quantizer_name]
    specialized_config.update(update_keys)
    return specialized_config


def clone_qconfig(config: AttrDict, **updates: Any) -> AttrDict:
    cloned = AttrDict(copy.deepcopy(config))
    cloned.update(updates)
    return cloned


class QConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | int,
        stride: tuple[int, int] | int,
        padding: tuple[int, int] | int,
        output_padding: tuple[int, int] | int,
        groups: int,
        bias: bool,
        dilation: tuple[int, int] | int,
        padding_mode: str,
        w_qconfig: AttrDict,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input: torch.Tensor, gamma: torch.Tensor | None = None) -> torch.Tensor:
        del gamma
        return F.conv_transpose2d(
            input,
            self.weight_fake_quant(self.weight),
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


def quantize_conv_transpose(module: nn.ConvTranspose2d, w_qconfig: AttrDict) -> QConvTranspose2d:
    qmodule = QConvTranspose2d(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        output_padding=module.output_padding,
        groups=module.groups,
        bias=module.bias is not None,
        dilation=module.dilation,
        padding_mode=module.padding_mode,
        w_qconfig=clone_qconfig(w_qconfig, ch_axis=1),
    )
    qmodule.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        qmodule.bias.data.copy_(module.bias.data)
    return qmodule


class QuantPreConvTransposeLayer(nn.Module):
    def __init__(self, module: nn.ConvTranspose2d, w_qconfig: AttrDict, a_qconfig: AttrDict) -> None:
        super().__init__()
        self.module = quantize_conv_transpose(module, w_qconfig)
        self.layer_pre_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_pre_act_fake_quantize(x)
        return self.module(x)


class QuantMLPBlock(QuantizedBlock):
    def __init__(self, org_module: nn.Module, w_qconfig: AttrDict, a_qconfig: AttrDict) -> None:
        super().__init__()
        self.lin1 = PreQuantizedLayer(org_module.lin1, None, w_qconfig, a_qconfig)
        self.lin2 = PreQuantizedLayer(org_module.lin2, None, w_qconfig, a_qconfig)
        self.act = org_module.act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class QuantDecoderAttentionBlock(QuantizedBlock):
    def __init__(
        self,
        org_module: Attention,
        w_qconfig: AttrDict,
        a_qconfig: AttrDict,
        ptq4sam_config: AttrDict,
    ) -> None:
        super().__init__()
        self.embedding_dim = org_module.embedding_dim
        self.internal_dim = org_module.internal_dim
        self.num_heads = org_module.num_heads

        self.q_proj = PreQuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig)
        self.k_proj = PreQuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig)
        self.v_proj = PreQuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig)
        self.out_proj = PreQuantizedLayer(org_module.out_proj, None, w_qconfig, a_qconfig)

        if ptq4sam_config.AGQ:
            softmax_a_config = update_specialized_quantizer_config(a_qconfig, "softmax")
        else:
            softmax_a_config = a_qconfig

        if ptq4sam_config.BIG:
            sign_a_config = update_specialized_quantizer_config(a_qconfig, "bimodal")
        else:
            sign_a_config = a_qconfig

        self.softmax_post_act_fake_quantize = Quantizer(None, softmax_a_config)
        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, sign_a_config)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)

        if ptq4sam_config.BIG:
            self.k_post_act_fake_quantize.global_num = ptq4sam_config.global_num
            self.k_post_act_fake_quantize.peak_distance = ptq4sam_config.peak_distance
            self.k_post_act_fake_quantize.peak_height = ptq4sam_config.peak_height

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        x = x.reshape(batch, tokens, num_heads, channels // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(batch, n_tokens, n_heads * c_per_head)

    def forward(self, qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        q, k, v = qkv

        q = self.q_post_act_fake_quantize(self.q_proj(q))
        k = self.k_post_act_fake_quantize(self.k_proj(k))
        v = self.v_post_act_fake_quantize(self.v_proj(v))

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / (c_per_head ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.softmax_post_act_fake_quantize(attn, value=v)

        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out

    def bimodal_adjust(self) -> None:
        if not getattr(self.k_post_act_fake_quantize, "is_bimodal", False):
            return
        sign = self.k_post_act_fake_quantize.sign

        def adjust_linear(linear: nn.Linear, sign_tensor: torch.Tensor) -> None:
            linear.weight.mul_(sign_tensor.unsqueeze(1))
            linear.bias.mul_(sign_tensor)

        adjust_linear(self.k_proj.module, sign)
        adjust_linear(self.q_proj.module, sign)
        self.k_post_act_fake_quantize.is_bimodal = False


class QuantGenericMLP(nn.Module):
    def __init__(self, org_module: nn.Module, w_qconfig: AttrDict, a_qconfig: AttrDict) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            PreQuantizedLayer(layer, None, w_qconfig, a_qconfig) for layer in org_module.layers
        )
        self.sigmoid_output = getattr(org_module, "sigmoid_output", False)
        self.output_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = torch.relu(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return self.output_post_act_fake_quantize(x)


class QuantDecoderTwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        org_module: TwoWayAttentionBlock,
        w_qconfig: AttrDict,
        a_qconfig: AttrDict,
        ptq4sam_config: AttrDict,
    ) -> None:
        super().__init__()
        self.self_attn = QuantDecoderAttentionBlock(
            org_module.self_attn, w_qconfig, a_qconfig, ptq4sam_config
        )
        self.norm1 = org_module.norm1
        self.cross_attn_token_to_image = QuantDecoderAttentionBlock(
            org_module.cross_attn_token_to_image, w_qconfig, a_qconfig, ptq4sam_config
        )
        self.norm2 = org_module.norm2
        self.mlp = QuantMLPBlock(org_module.mlp, w_qconfig, a_qconfig)
        self.norm3 = org_module.norm3
        self.norm4 = org_module.norm4
        self.cross_attn_image_to_token = QuantDecoderAttentionBlock(
            org_module.cross_attn_image_to_token, w_qconfig, a_qconfig, ptq4sam_config
        )
        self.skip_first_layer_pe = org_module.skip_first_layer_pe

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: torch.Tensor,
        key_pe: torch.Tensor,
        kd_targets=None,
        layer_idx=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn((queries, queries, queries))
        else:
            q = queries + query_pe
            attn_out = self.self_attn((q, q, queries))
            queries = queries + attn_out
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image((q, k, keys))
        queries = queries + attn_out
        queries = self.norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token((k, q, queries))
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class QuantEdgeTwoWayTransformer(nn.Module):
    def __init__(
        self,
        org_module: TwoWayTransformer,
        w_qconfig: AttrDict,
        a_qconfig: AttrDict,
        ptq4sam_config: AttrDict,
    ) -> None:
        super().__init__()
        self.depth = org_module.depth
        self.embedding_dim = org_module.embedding_dim
        self.num_heads = org_module.num_heads
        self.mlp_dim = org_module.mlp_dim
        self.layers = nn.ModuleList(
            QuantDecoderTwoWayAttentionBlock(layer, w_qconfig, a_qconfig, ptq4sam_config)
            for layer in org_module.layers
        )
        self.final_attn_token_to_image = QuantDecoderAttentionBlock(
            org_module.final_attn_token_to_image, w_qconfig, a_qconfig, ptq4sam_config
        )
        self.norm_final_attn = org_module.norm_final_attn

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
        kd_targets=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for idx, layer in enumerate(self.layers):
            queries, keys = layer(queries, keys, point_embedding, image_pe, kd_targets, idx)

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image((q, k, keys))
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class QuantDecoderLabelEmbedding(nn.Module):
    def __init__(self, org_module: DecoderLabelEmbedding, w_qconfig: AttrDict, a_qconfig: AttrDict) -> None:
        super().__init__()
        self.point_embeddings = org_module.point_embeddings
        self.not_a_point_embed = org_module.not_a_point_embed
        self.num_point_embeddings = org_module.num_point_embeddings
        self.input_fake_quantize = Quantizer(None, a_qconfig)
        self.not_a_point_weight_fake_quantize = WeightQuantizer(clone_qconfig(w_qconfig))
        self.point_weight_fake_quantizers = nn.ModuleList(
            WeightQuantizer(clone_qconfig(w_qconfig)) for _ in range(self.num_point_embeddings)
        )
        self.output_fake_quantize = Quantizer(None, a_qconfig)

    @staticmethod
    def _float_eq(x: torch.Tensor, value: float) -> torch.Tensor:
        diff = x - value
        return torch.relu(1.0 - diff * diff)

    def forward(self, point_embedding_pe: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_embedding_pe = self.input_fake_quantize(point_embedding_pe)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding_pe)
        mask_neg1 = self._float_eq(point_labels, -1.0)
        sparse_embedding = point_embedding_pe * (1.0 - mask_neg1)
        sparse_embedding = sparse_embedding + self.not_a_point_weight_fake_quantize(
            self.not_a_point_embed.weight
        ) * mask_neg1

        for index in range(self.num_point_embeddings):
            mask = self._float_eq(point_labels, float(index))
            sparse_embedding = sparse_embedding + self.point_weight_fake_quantizers[index](
                self.point_embeddings[index].weight
            ) * mask
        return self.output_fake_quantize(sparse_embedding)


class QuantOutputUpscaling(nn.Module):
    def __init__(self, org_module: nn.Sequential, w_qconfig: AttrDict, a_qconfig: AttrDict) -> None:
        super().__init__()
        self.deconv0 = QuantPreConvTransposeLayer(org_module[0], w_qconfig, a_qconfig)
        self.norm = org_module[1]
        self.act0 = org_module[2]
        self.deconv1 = QuantPreConvTransposeLayer(org_module[3], w_qconfig, a_qconfig)
        self.act1 = org_module[4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv0(x)
        x = self.norm(x)
        x = self.act0(x)
        x = self.deconv1(x)
        x = self.act1(x)
        return x


class QuantDecoderHypernetworkStack(nn.Module):
    def __init__(self, org_module: DecoderHypernetworkStack, w_qconfig: AttrDict, a_qconfig: AttrDict) -> None:
        super().__init__()
        self.mlps = nn.ModuleList(
            QuantGenericMLP(mlp, w_qconfig, a_qconfig) for mlp in org_module.mlps
        )
        self.num_mask_tokens = org_module.num_mask_tokens

    def forward(self, mask_tokens_out: torch.Tensor) -> torch.Tensor:
        hyper_in_list = []
        batch_size, _, embedding_dim = mask_tokens_out.shape
        for index in range(self.num_mask_tokens):
            token = mask_tokens_out[:, index : index + 1, :].reshape(batch_size, embedding_dim)
            hyper_in_list.append(self.mlps[index](token))
        return torch.stack(hyper_in_list, dim=1)


class QuantMaskProjection(nn.Module):
    def __init__(self, a_qconfig: AttrDict) -> None:
        super().__init__()
        self.matmul = QuantizedMatMul(a_qconfig)

    def forward(self, hyper_in: torch.Tensor, upscaled: torch.Tensor) -> torch.Tensor:
        batch_size = upscaled.size(0)
        channels = upscaled.size(1)
        spatial = upscaled.size(2) * upscaled.size(3)
        flat = upscaled.reshape(batch_size, channels, spatial)
        return self.matmul((hyper_in, flat))


class QuantDecoderScoreHead(nn.Module):
    def __init__(self, org_module: DecoderScoreHead, w_qconfig: AttrDict, a_qconfig: AttrDict) -> None:
        super().__init__()
        self.use_stability_score = org_module.use_stability_score
        self.mask_threshold = org_module.mask_threshold
        self.stability_score_offset = org_module.stability_score_offset
        self.mask_pre_act_fake_quantize = Quantizer(None, a_qconfig)
        self.score_post_act_fake_quantize = Quantizer(None, a_qconfig)
        if not self.use_stability_score:
            self.iou_prediction_head = QuantGenericMLP(org_module.iou_prediction_head, w_qconfig, a_qconfig)
        else:
            self.iou_prediction_head = org_module.iou_prediction_head

    @staticmethod
    def _stability_score_npu(masks: torch.Tensor, mask_threshold: float, threshold_offset: float) -> torch.Tensor:
        k = 50.0
        high = torch.sigmoid(k * (masks - (mask_threshold + threshold_offset)))
        low = torch.sigmoid(k * (masks - (mask_threshold - threshold_offset)))
        intersections = high.sum(-1).sum(-1)
        unions = low.sum(-1).sum(-1)
        return intersections / unions

    def forward(self, iou_token_out: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if self.use_stability_score:
            scores = self._stability_score_npu(
                self.mask_pre_act_fake_quantize(masks),
                self.mask_threshold,
                self.stability_score_offset,
            )
        else:
            scores = self.iou_prediction_head(iou_token_out)
        return self.score_post_act_fake_quantize(scores)


class QuantFullDecoderSurface(nn.Module):
    def __init__(self, org_module: nn.Module, config_quant: AttrDict) -> None:
        super().__init__()
        self.label_embedding = QuantDecoderLabelEmbedding(
            org_module.label_embedding,
            config_quant.w_qconfig,
            config_quant.a_qconfig,
        )
        self.transformer = QuantEdgeTwoWayTransformer(
            org_module.transformer,
            config_quant.w_qconfig,
            config_quant.a_qconfig,
            config_quant.ptq4sam,
        )
        self.iou_token = org_module.iou_token
        self.mask_tokens = org_module.mask_tokens
        self.iou_token_weight_fake_quantize = WeightQuantizer(clone_qconfig(config_quant.w_qconfig))
        self.mask_tokens_weight_fake_quantize = WeightQuantizer(clone_qconfig(config_quant.w_qconfig))
        self.register_buffer("dense_embedding", org_module.dense_embedding.clone())
        self.register_buffer("image_pe", org_module.image_pe.clone())
        self.output_upscaling = QuantOutputUpscaling(
            org_module.output_upscaling,
            config_quant.w_qconfig,
            config_quant.a_qconfig,
        )
        self.output_hypernetworks = QuantDecoderHypernetworkStack(
            org_module.output_hypernetworks,
            config_quant.w_qconfig,
            config_quant.a_qconfig,
        )
        self.mask_projection = QuantMaskProjection(config_quant.a_qconfig)
        self.score_head = QuantDecoderScoreHead(
            org_module.score_head,
            config_quant.w_qconfig,
            config_quant.a_qconfig,
        )
        self.num_mask_tokens = org_module.num_mask_tokens
        self.embed_h = org_module.embed_h
        self.embed_w = org_module.embed_w

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_embedding_pe: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_embedding = self.label_embedding(point_embedding_pe, point_labels)

        iou_token = self.iou_token_weight_fake_quantize(self.iou_token.weight)
        mask_tokens = self.mask_tokens_weight_fake_quantize(self.mask_tokens.weight)
        output_tokens = torch.cat([iou_token, mask_tokens], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_embedding.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_embedding), dim=1)

        src = image_embeddings + self.dense_embedding
        hs, src = self.transformer(src, self.image_pe, tokens)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        batch_size = src.size(0)
        embed_dim = src.size(2)
        src = src.transpose(1, 2).reshape(batch_size, embed_dim, self.embed_h, self.embed_w)

        upscaled = self.output_upscaling(src)
        hyper_in = self.output_hypernetworks(mask_tokens_out)
        masks = self.mask_projection(hyper_in, upscaled).reshape(
            batch_size,
            -1,
            upscaled.size(2),
            upscaled.size(3),
        )
        scores = self.score_head(iou_token_out, masks)
        return scores, masks


def bimodal_adjust(model: nn.Module) -> None:
    LOGGER.info("start bimodal integration")
    for name, module in model.named_modules():
        if isinstance(module, QuantDecoderAttentionBlock) and "token_to_image" in name:
            LOGGER.info("bimodal probe on %s", name)
            module.bimodal_adjust()
    LOGGER.info("bimodal integration end")


def quantize_decoder_surface(
    model: nn.Module,
    config_quant: AttrDict,
    scope: str = "transformer",
) -> nn.Module:
    if scope == "full":
        return QuantFullDecoderSurface(model, config_quant)
    if scope != "transformer":
        raise ValueError(f"Unsupported quantization scope: {scope}")
    model.transformer = QuantEdgeTwoWayTransformer(
        model.transformer,
        config_quant.w_qconfig,
        config_quant.a_qconfig,
        config_quant.ptq4sam,
    )
    return model


@torch.no_grad()
def calibrate_decoder(
    model: nn.Module,
    calibration_samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    big: bool,
) -> None:
    if big:
        model(*calibration_samples[0])
        bimodal_adjust(model)

    enable_calibration_woquantization(model, quantizer_type="act_fake_quant")
    for sample in calibration_samples:
        model(*sample)

    enable_calibration_woquantization(model, quantizer_type="weight_fake_quant")
    model(*calibration_samples[0])


@dataclass
class EvalBundle:
    mean_metrics: dict[str, float]
    per_sample: list[dict[str, Any]]


def evaluate_decoder(
    fp_model: nn.Module,
    quant_model: nn.Module,
    eval_paths: list[tuple[Path, Path, Path]],
) -> EvalBundle:
    device = next(fp_model.parameters()).device
    eval_samples = [tuple(tensor.to(device) for tensor in load_decoder_sample_triplet(paths)) for paths in eval_paths]
    fp_outputs = _run_decoder_model(fp_model, eval_samples)
    quant_outputs = _run_decoder_model(quant_model, eval_samples)

    per_sample: list[dict[str, Any]] = []
    finite_metrics: dict[str, list[float]] = {}
    for paths, ref_output, quant_output in zip(eval_paths, fp_outputs, quant_outputs):
        metrics = compute_decoder_metrics(ref_output, quant_output)
        metrics["sample"] = paths[0].name
        for key, value in list(metrics.items()):
            if key == "sample":
                continue
            value = float(value)
            metrics[key] = value
            if math.isfinite(value):
                finite_metrics.setdefault(key, []).append(value)
            else:
                LOGGER.warning(
                    "Non-finite metric detected and excluded from mean: sample=%s metric=%s value=%s",
                    metrics["sample"],
                    key,
                    value,
                )
        per_sample.append(metrics)

    metric_keys = [key for key in per_sample[0].keys() if key != "sample"]
    mean_metrics: dict[str, float] = {}
    for key in metric_keys:
        values = finite_metrics.get(key, [])
        if values:
            mean_metrics[key] = float(sum(values) / len(values))
        else:
            LOGGER.warning("All values are non-finite for metric=%s; mean will be NaN", key)
            mean_metrics[key] = float("nan")
    return EvalBundle(mean_metrics=mean_metrics, per_sample=per_sample)


def move_samples_to_device(
    sample_paths: list[tuple[Path, Path, Path]],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    samples = [load_decoder_sample_triplet(paths) for paths in sample_paths]
    return [tuple(tensor.to(device) for tensor in sample) for sample in samples]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PTQ4SAM-style uint8 PTQ for EdgeSAM decoder only")
    parser.add_argument(
        "--checkpoint",
        default="/home/kitemanul/project/EdgeSAM/weights/edge_sam.pth",
        help="Path to EdgeSAM checkpoint (.pth)",
    )
    parser.add_argument(
        "--calibration-list",
        default="/home/kitemanul/project/EdgeSAM/weights/one_pipeline/calib/decoder_datalist_20.txt",
        help="Decoder calibration datalist",
    )
    parser.add_argument(
        "--eval-list",
        default="/home/kitemanul/project/EdgeSAM/weights/one_pipeline/calib/decoder_test_5.txt",
        help="Decoder evaluation datalist",
    )
    parser.add_argument("--calibration-count", type=int, default=20)
    parser.add_argument("--eval-count", type=int, default=5)
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument(
        "--scope",
        choices=("transformer", "full"),
        default="transformer",
        help="Quantization boundary: transformer-only or the full NPU-safe decoder surface",
    )
    parser.add_argument("--disable-big", action="store_true")
    parser.add_argument("--disable-agq", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save summary.json; defaults to PTQ4SAM/results/<timestamp>",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    setup_logging()
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because PTQ4SAM BIG/AGQ fake-quant code uses CUDA tensors")

    device = torch.device("cuda:0")
    checkpoint_path = Path(args.checkpoint).resolve()
    calibration_paths = collect_decoder_sample_triplets(args.calibration_list, limit=args.calibration_count)
    eval_paths = collect_decoder_sample_triplets(args.eval_list, limit=args.eval_count)

    if len(calibration_paths) < args.calibration_count:
        raise ValueError(f"Requested {args.calibration_count} calibration samples but found {len(calibration_paths)}")
    if len(eval_paths) < args.eval_count:
        raise ValueError(f"Requested {args.eval_count} eval samples but found {len(eval_paths)}")

    config_quant = make_qconfig(bit=args.bit)
    config_quant.ptq4sam.BIG = not args.disable_big
    config_quant.ptq4sam.AGQ = not args.disable_agq

    LOGGER.info("loading fp32 decoder surface from %s", checkpoint_path)
    fp_model = _build_decoder_surface(checkpoint_path, use_stability_score=True).to(device).eval()

    LOGGER.info("building quantized decoder surface with scope=%s", args.scope)
    quant_model = _build_decoder_surface(checkpoint_path, use_stability_score=True).to(device).eval()
    quant_model = quantize_decoder_surface(quant_model, config_quant, scope=args.scope)
    quant_model = quant_model.to(device).eval()

    calibration_samples = move_samples_to_device(calibration_paths, device)
    LOGGER.info("running calibration on %d samples", len(calibration_samples))
    calibrate_decoder(quant_model, calibration_samples, config_quant.ptq4sam.BIG)
    enable_quantization(quant_model)

    LOGGER.info("running fp32 vs uint8 comparison on %d samples", len(eval_paths))
    bundle = evaluate_decoder(fp_model, quant_model, eval_paths)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path("/home/kitemanul/project/PTQ4SAM/results")
            / "edgesam_decoder_ptq4sam_uint8"
            / args.scope
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(checkpoint_path),
        "calibration_list": str(Path(args.calibration_list).resolve()),
        "eval_list": str(Path(args.eval_list).resolve()),
        "calibration_count": args.calibration_count,
        "eval_count": args.eval_count,
        "bit": args.bit,
        "scope": args.scope,
        "big": config_quant.ptq4sam.BIG,
        "agq": config_quant.ptq4sam.AGQ,
        "mean_metrics": bundle.mean_metrics,
        "per_sample": bundle.per_sample,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary["mean_metrics"], indent=2))
    print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()
