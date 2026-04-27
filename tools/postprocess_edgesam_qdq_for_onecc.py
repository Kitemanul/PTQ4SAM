#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

NP_DTYPE = {
    TensorProto.FLOAT: np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT64: np.int64,
    TensorProto.INT32: np.int32,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT8: np.int8,
    TensorProto.UINT16: np.uint16,
    TensorProto.INT16: np.int16,
    TensorProto.BOOL: np.bool_,
}


def attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for item in node.attribute:
        if item.name == name:
            return helper.get_attribute_value(item)
    return default


def safe(name: str, fallback: str) -> str:
    return name.strip('/').replace('/', '_') or fallback


def replace_nodes(model: onnx.ModelProto, nodes: list[onnx.NodeProto]) -> None:
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def read_shape(value_info: onnx.ValueInfoProto) -> list[int] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField('shape'):
        return None
    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField('dim_value'):
            return None
        dims.append(int(dim.dim_value))
    return dims


def broadcast(lhs: list[int], rhs: list[int]) -> list[int] | None:
    out: list[int] = []
    for a, b in zip(reversed(lhs), reversed(rhs)):
        if a == b:
            out.append(a)
        elif a == 1:
            out.append(b)
        elif b == 1:
            out.append(a)
        else:
            return None
    longer = lhs if len(lhs) > len(rhs) else rhs
    out.extend(reversed(longer[: abs(len(lhs) - len(rhs))]))
    return list(reversed(out))


def slice_array(data: np.ndarray, starts: np.ndarray, ends: np.ndarray, axes: np.ndarray | None, steps: np.ndarray | None) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64).flatten()
    ends = np.asarray(ends, dtype=np.int64).flatten()
    axes = np.arange(data.ndim, dtype=np.int64) if axes is None else np.asarray(axes, dtype=np.int64).flatten()
    steps = np.ones_like(starts, dtype=np.int64) if steps is None else np.asarray(steps, dtype=np.int64).flatten()
    slices: list[slice] = [slice(None)] * data.ndim
    for start, end, axis, step in zip(starts, ends, axes, steps):
        axis = int(axis)
        dim = data.shape[axis]
        if end > dim:
            end = dim
        if end < -dim:
            end = -dim
        slices[axis] = slice(int(start), int(end), int(step))
    return data[tuple(slices)]


def resolve_reshape(target: np.ndarray, input_shape: list[int] | None) -> list[int]:
    out = [int(x) for x in np.asarray(target).flatten()]
    if input_shape is None or any(dim <= 0 for dim in input_shape):
        return out
    resolved: list[int] = []
    unknown = -1
    known_product = 1
    for index, dim in enumerate(out):
        if dim == 0:
            dim = int(input_shape[index])
        if dim == -1:
            unknown = index
            resolved.append(-1)
        else:
            known_product *= int(dim)
            resolved.append(int(dim))
    if unknown >= 0:
        resolved[unknown] = int(np.prod(input_shape)) // known_product
    return resolved


class Postprocessor:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        self.constants: dict[str, np.ndarray] = {}
        self.shapes: dict[str, list[int]] = {}
        self.initializer_names = {item.name for item in model.graph.initializer}
        self.added_initializers: list[onnx.TensorProto] = []
        self.summary = {
            'reshape_shapes_staticized': 0,
            'slice_params_staticized': 0,
            'less_rewritten': 0,
            'greater_equal_rewritten': 0,
            'less_equal_rewritten': 0,
            'dynamic_pow_rewritten': 0,
            'clip_rewritten': 0,
            'constant_cast_folded': 0,
            'dead_nodes_pruned': 0,
        }
        for initializer in model.graph.initializer:
            array = numpy_helper.to_array(initializer)
            self.constants[initializer.name] = array
            self.shapes[initializer.name] = list(array.shape)
        for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
            shape = read_shape(value_info)
            if shape is not None:
                self.shapes[value_info.name] = shape

    def add_initializer(self, name: str, array: np.ndarray) -> str:
        if name not in self.initializer_names:
            array = np.asarray(array)
            self.added_initializers.append(numpy_helper.from_array(array, name))
            self.initializer_names.add(name)
            self.constants[name] = array
            self.shapes[name] = list(array.shape)
        return name

    def propagate(self) -> None:
        for node in self.model.graph.node:
            if node.op_type == 'Constant':
                for item in node.attribute:
                    if item.name == 'value':
                        array = numpy_helper.to_array(item.t)
                        self.constants[node.output[0]] = array
                        self.shapes[node.output[0]] = list(array.shape)
            elif node.op_type == 'Shape' and node.input[0] in self.shapes:
                array = np.asarray(self.shapes[node.input[0]], dtype=np.int64)
                self.constants[node.output[0]] = array
                self.shapes[node.output[0]] = list(array.shape)
            elif node.op_type == 'Slice' and node.input[0] in self.constants and node.input[1] in self.constants and node.input[2] in self.constants:
                axes = self.constants.get(node.input[3]) if len(node.input) > 3 and node.input[3] else None
                steps = self.constants.get(node.input[4]) if len(node.input) > 4 and node.input[4] else None
                array = slice_array(self.constants[node.input[0]], self.constants[node.input[1]], self.constants[node.input[2]], axes, steps)
                self.constants[node.output[0]] = array
                self.shapes[node.output[0]] = list(array.shape)
            elif node.op_type == 'Unsqueeze' and node.input[0] in self.constants:
                axes = attr(node, 'axes')
                if axes is None and len(node.input) > 1 and node.input[1] in self.constants:
                    axes = np.asarray(self.constants[node.input[1]], dtype=np.int64).flatten().tolist()
                if axes is not None:
                    array = self.constants[node.input[0]]
                    for axis in sorted(int(axis) for axis in axes):
                        array = np.expand_dims(array, axis)
                    self.constants[node.output[0]] = array
                    self.shapes[node.output[0]] = list(array.shape)
            elif node.op_type == 'Concat' and all(name in self.constants for name in node.input):
                array = np.concatenate([np.asarray(self.constants[name]) for name in node.input], axis=int(attr(node, 'axis', 0)))
                self.constants[node.output[0]] = array
                self.shapes[node.output[0]] = list(array.shape)
            elif node.op_type in {'Add', 'Sub', 'Mul', 'Div', 'Max', 'Min'} and node.input[0] in self.shapes and node.input[1] in self.shapes:
                shape = broadcast(self.shapes[node.input[0]], self.shapes[node.input[1]])
                if shape is not None:
                    self.shapes[node.output[0]] = shape
            elif node.op_type in {'QuantizeLinear', 'DequantizeLinear'} and node.input[0] in self.shapes:
                self.shapes[node.output[0]] = self.shapes[node.input[0]]
            elif node.op_type == 'Reshape' and len(node.input) >= 2 and node.input[1] in self.constants:
                self.shapes[node.output[0]] = resolve_reshape(self.constants[node.input[1]], self.shapes.get(node.input[0]))
            elif node.op_type == 'Transpose' and node.input[0] in self.shapes:
                perm = attr(node, 'perm')
                if perm is not None:
                    self.shapes[node.output[0]] = [self.shapes[node.input[0]][int(index)] for index in perm]


    def staticize_reshape_shapes(self) -> None:
        self.propagate()
        for node in self.model.graph.node:
            if node.op_type != 'Reshape' or len(node.input) < 2 or node.input[1] not in self.constants:
                continue
            shape = np.asarray(resolve_reshape(self.constants[node.input[1]], self.shapes.get(node.input[0])), dtype=np.int64)
            node.input[1] = self.add_initializer(f'{safe(node.name, 'reshape')}_static_shape', shape)
            self.shapes[node.output[0]] = shape.astype(int).tolist()
            self.summary['reshape_shapes_staticized'] += 1

    def staticize_slice_params(self) -> None:
        self.propagate()
        for node in self.model.graph.node:
            if node.op_type != 'Slice':
                continue
            for index in range(1, len(node.input)):
                old_input = node.input[index]
                if old_input and old_input in self.constants:
                    node.input[index] = self.add_initializer(
                        f'{safe(node.name, 'slice')}_param_{index}',
                        np.asarray(self.constants[old_input], dtype=np.int64),
                    )
                    self.summary['slice_params_staticized'] += 1

    def rewrite_boolean_ops(self) -> None:
        nodes: list[onnx.NodeProto] = []
        for node in self.model.graph.node:
            if node.op_type == 'Less':
                node.op_type = 'Greater'
                node.input[0], node.input[1] = node.input[1], node.input[0]
                self.summary['less_rewritten'] += 1
                nodes.append(node)
            elif node.op_type == 'GreaterOrEqual':
                tmp = f'{node.output[0]}_greater_swapped'
                nodes.append(helper.make_node('Greater', [node.input[1], node.input[0]], [tmp], name=f'{node.name}_as_greater'))
                nodes.append(helper.make_node('Not', [tmp], [node.output[0]], name=f'{node.name}_as_not'))
                self.summary['greater_equal_rewritten'] += 1
            elif node.op_type == 'LessOrEqual':
                tmp = f'{node.output[0]}_greater'
                nodes.append(helper.make_node('Greater', [node.input[0], node.input[1]], [tmp], name=f'{node.name}_as_greater'))
                nodes.append(helper.make_node('Not', [tmp], [node.output[0]], name=f'{node.name}_as_not'))
                self.summary['less_equal_rewritten'] += 1
            else:
                nodes.append(node)
        replace_nodes(self.model, nodes)

    def rewrite_dynamic_pow(self) -> None:
        self.propagate()
        ln2 = self.add_initializer('__const_ln2_f32', np.asarray(math.log(2.0), dtype=np.float32))
        nodes: list[onnx.NodeProto] = []
        for node in self.model.graph.node:
            is_pow2 = (
                node.op_type == 'Pow'
                and node.input[0] in self.constants
                and np.asarray(self.constants[node.input[0]]).shape == ()
                and abs(float(np.asarray(self.constants[node.input[0]])) - 2.0) < 1e-6
                and node.input[1] not in self.constants
            )
            if is_pow2:
                mul_out = f'{node.output[0]}_mul_ln2'
                nodes.append(helper.make_node('Mul', [node.input[1], ln2], [mul_out], name=f'{node.name}_mul_ln2'))
                nodes.append(helper.make_node('Exp', [mul_out], [node.output[0]], name=f'{node.name}_as_exp'))
                self.summary['dynamic_pow_rewritten'] += 1
            else:
                nodes.append(node)
        replace_nodes(self.model, nodes)

    def rewrite_clip(self) -> None:
        nodes: list[onnx.NodeProto] = []
        for node in self.model.graph.node:
            if node.op_type != 'Clip':
                nodes.append(node)
                continue
            x = node.input[0]
            min_input = node.input[1] if len(node.input) > 1 else ''
            max_input = node.input[2] if len(node.input) > 2 else ''
            if min_input and max_input:
                tmp = f'{node.output[0]}_clip_min'
                nodes.append(helper.make_node('Max', [x, min_input], [tmp], name=f'{node.name}_as_max'))
                nodes.append(helper.make_node('Min', [tmp, max_input], [node.output[0]], name=f'{node.name}_as_min'))
            elif min_input:
                nodes.append(helper.make_node('Max', [x, min_input], [node.output[0]], name=f'{node.name}_as_max'))
            elif max_input:
                nodes.append(helper.make_node('Min', [x, max_input], [node.output[0]], name=f'{node.name}_as_min'))
            else:
                nodes.append(helper.make_node('Identity', [x], [node.output[0]], name=f'{node.name}_as_identity'))
            self.summary['clip_rewritten'] += 1
        replace_nodes(self.model, nodes)


    def fold_constant_casts(self) -> None:
        self.propagate()
        replacements: dict[str, str] = {}
        nodes: list[onnx.NodeProto] = []
        for node in self.model.graph.node:
            for index, name in enumerate(node.input):
                if name in replacements:
                    node.input[index] = replacements[name]
            if node.op_type == 'Cast' and node.input[0] in self.constants:
                to_dtype = attr(node, 'to')
                if to_dtype in NP_DTYPE:
                    array = np.asarray(self.constants[node.input[0]]).astype(NP_DTYPE[to_dtype])
                    replacements[node.output[0]] = self.add_initializer(f'{safe(node.name, 'cast')}_folded', array)
                    self.summary['constant_cast_folded'] += 1
                    continue
            nodes.append(node)
        for node in nodes:
            for index, name in enumerate(node.input):
                if name in replacements:
                    node.input[index] = replacements[name]
        replace_nodes(self.model, nodes)

    def prune(self) -> None:
        needed = {output.name for output in self.model.graph.output}
        kept: list[onnx.NodeProto] = []
        original = len(self.model.graph.node)
        for node in reversed(self.model.graph.node):
            if any(output in needed for output in node.output):
                kept.append(node)
                needed.update(name for name in node.input if name)
        kept.reverse()
        replace_nodes(self.model, kept)
        self.summary['dead_nodes_pruned'] += original - len(kept)

    def run(self) -> onnx.ModelProto:
        self.staticize_reshape_shapes()
        self.staticize_slice_params()
        self.rewrite_boolean_ops()
        self.rewrite_dynamic_pow()
        self.rewrite_clip()
        self.fold_constant_casts()
        self.model.graph.initializer.extend(self.added_initializers)
        self.prune()
        inferred = onnx.shape_inference.infer_shapes(self.model)
        onnx.checker.check_model(inferred)
        return inferred


def postprocess(input_path: Path, output_path: Path) -> dict[str, int]:
    model = onnx.load(input_path)
    processor = Postprocessor(model)
    output_model = processor.run()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(output_model, output_path)
    return processor.summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Postprocess EdgeSAM Q/DQ ONNX for local ONE/circle-mlir.')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    summary = postprocess(args.input, args.output)
    print({'output': str(args.output), **summary})


if __name__ == '__main__':
    main()
