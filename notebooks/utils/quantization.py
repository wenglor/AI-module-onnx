import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from onnx import helper, numpy_helper, shape_inference
from onnxruntime.quantization.quantize import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class TorchCalibrationDataReader(CalibrationDataReader):
    def __init__(self, model_path, samples=500, **kwargs):
        """
        Initializes a PyTorch data reader for ONNX Runtime quantization.

        Args:
            model_path (str): Path to the ONNX model to be quantized.
            samples (int): The number of samples to iterate over for calibration. Defaults to 500.
        """
        session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.counter = 0
        self.samples = samples
        self.kwargs = kwargs
        self.input_name = session.get_inputs()[0].name
        self.dataloader = iter(torch.utils.data.DataLoader(**kwargs))

    def get_next(self):
        if self.counter < self.samples:
            inputs, *_ = next(self.dataloader)
            output = {self.input_name: inputs.cpu().numpy()}
        else:
            output = None
        self.counter += 1
        if self.counter % len(self.dataloader) == 0:
            self.dataloader = iter(torch.utils.data.DataLoader(**self.kwargs))

        return output


def find_postprocess_nodes_to_exclude(onnx_model_path):
    """
    Auto-discover post-processing node names to exclude from quantization.

    Traces backward from the NonMaxSuppression op, stopping at Conv/Sigmoid
    boundaries (head output nodes that should stay quantized). Also traces
    forward from NMS to include output-gathering nodes.

    This keeps the backbone+neck+head fully quantized (one large subgraph
    for NPU) while preserving fp32 precision for bbox decode math and NMS
    inputs, which is critical for correct NMS behavior.

    Args:
        onnx_model_path: Path to the preprocessed fp32 ONNX model.

    Returns:
        List of node names to pass to quantize_static(nodes_to_exclude=...).
        Returns empty list if no NonMaxSuppression op is found.
    """
    from collections import defaultdict

    model = onnx.load(str(onnx_model_path))
    nodes = list(model.graph.node)

    # Map output tensor → producing node index
    out_to_idx = {}
    for i, node in enumerate(nodes):
        for o in node.output:
            out_to_idx[o] = i

    # Map input tensor → consuming node indices
    inp_to_nodes = defaultdict(list)
    for i, node in enumerate(nodes):
        for inp in node.input:
            inp_to_nodes[inp].append(i)

    # Find NMS node
    nms_idx = None
    for i, n in enumerate(nodes):
        if n.op_type == "NonMaxSuppression":
            nms_idx = i
            break
    if nms_idx is None:
        return []

    visited = set()
    exclude_names = []

    # Include the NMS node itself
    if nodes[nms_idx].name:
        exclude_names.append(nodes[nms_idx].name)
    visited.add(nms_idx)

    # BFS backward from NMS inputs, stop at Conv/Sigmoid boundaries
    stop_types = {"Conv", "Sigmoid"}
    queue = list(nodes[nms_idx].input)
    while queue:
        tensor = queue.pop(0)
        if tensor not in out_to_idx:
            continue
        idx = out_to_idx[tensor]
        if idx in visited:
            continue
        visited.add(idx)
        node = nodes[idx]
        if node.op_type in stop_types:
            continue
        if node.name:
            exclude_names.append(node.name)
        for inp in node.input:
            queue.append(inp)

    # BFS forward from NMS outputs (output-gathering nodes)
    fwd_queue = []
    for o in nodes[nms_idx].output:
        for consumer_idx in inp_to_nodes.get(o, []):
            fwd_queue.append(consumer_idx)
    while fwd_queue:
        idx = fwd_queue.pop(0)
        if idx in visited:
            continue
        visited.add(idx)
        node = nodes[idx]
        if node.name:
            exclude_names.append(node.name)
        for o in node.output:
            for consumer_idx in inp_to_nodes.get(o, []):
                fwd_queue.append(consumer_idx)

    return exclude_names


def get_nodes_to_exclude(onnx_model):
    """Finds the node names of first conv, last gemm and the output activation
    (softmax, or sigmoid for multi-label models).
    Excluding these nodes is a best practice for minimizing quantization degradation.
    Excluding the output activation also keeps the classification head in one
    unquantized part, which the heatmap computation relies on."""

    all_nodes = onnx_model.graph.node
    graph_outputs = {output.name for output in onnx_model.graph.output}
    first_conv_name = next(
        (node.name for node in all_nodes if node.op_type == "Conv"), None
    )
    gemm_nodes = [node.name for node in all_nodes if node.op_type == "Gemm"]
    last_gemm_name = gemm_nodes[-1] if gemm_nodes else None

    nodes_to_exclude = [
        node.name
        for node in onnx_model.graph.node
        if "Softmax" in node.name
        or (
            node.op_type in ("Softmax", "Sigmoid")
            and graph_outputs.intersection(node.output)
        )
        or node.name == first_conv_name
        or node.name == last_gemm_name
    ]
    return nodes_to_exclude


def sort_nodes_topologically(model: onnx.ModelProto):
    """Reorder graph nodes into "latest-possible" topological order.

    Args:
        model: the loaded ONNX model to reorder.

    Returns:
        the same model with graph.node reordered in-place.
    """
    graph = model.graph

    # Map each output tensor to the node that produces it.
    producer = {out: node for node in graph.node for out in node.output}

    # For each node, count how many of its outputs are consumed by other nodes
    # (i.e. reverse out-degree).  Outputs that feed into graph outputs or are
    # initializer-produced are treated as already "consumed".
    graph_output_names = {o.name for o in graph.output}
    initializer_names = {i.name for i in graph.initializer}
    graph_input_names = {i.name for i in graph.input}
    external = graph_output_names | initializer_names | graph_input_names

    # pending[node_id] = number of this node's outputs still waiting to be scheduled.
    # A node is "ready" (in reverse order) when pending reaches 0.
    consumers: dict = {id(n): set() for n in graph.node}
    for node in graph.node:
        for inp in node.input:
            if inp and inp not in external and inp in producer:
                prod = producer[inp]
                consumers[id(prod)].add(id(node))

    # Reverse Kahn's: start from nodes whose outputs are only consumed by external
    # sinks (i.e. graph outputs or nothing).
    pending = {id(n): len(consumers[id(n)]) for n in graph.node}
    node_by_id = {id(n): n for n in graph.node}

    stack = [nid for nid, cnt in pending.items() if cnt == 0]
    reverse_order = []

    while stack:
        nid = stack.pop()
        node = node_by_id[nid]
        reverse_order.append(node)
        seen_producers: set = set()
        for inp in node.input:
            if inp and inp not in external and inp in producer:
                prod_id = id(producer[inp])
                if prod_id not in seen_producers:
                    seen_producers.add(prod_id)
                    pending[prod_id] -= 1
                    if pending[prod_id] == 0:
                        stack.append(prod_id)

    if len(reverse_order) != len(graph.node):
        raise RuntimeError(
            f"sort_nodes_topologically: only sorted {len(reverse_order)} of "
            f"{len(graph.node)} nodes — the graph may contain a cycle."
        )

    del graph.node[:]
    graph.node.extend(reversed(reverse_order))


_CPU_FALLBACK_PREFIX = "cpu_fallback"
# Cheap nodes that are moved to the CPU when only unquantized nodes follow them.
_CHEAP_TAIL_OPS = {
    "Concat",
    "Flatten",
    "Gather",
    "Identity",
    "Reshape",
    "Slice",
    "Split",
    "Squeeze",
    "Transpose",
    "Unsqueeze",
    "Clip",
    "HardSigmoid",
    "LeakyRelu",
    "Relu",
    "Sigmoid",
    "Tanh",
}


def keep_fp32_nodes_on_cpu(model: onnx.ModelProto) -> onnx.ModelProto:
    """Keeps the unquantized nodes of a QDQ model on the CPU.

    Nodes excluded from quantization (e.g. the first and last layers of a classifier, see
    get_nodes_to_exclude, or the post-processing of a detector, see
    find_postprocess_nodes_to_exclude) run faster on the CPU. This inserts a few
    lightweight nodes where the model switches between quantized and unquantized parts,
    so the unquantized parts are executed on the CPU while the quantized parts stay
    hardware-accelerated. The model's inputs, outputs and results are unchanged.

    Args:
        model: a single-input QDQ model with static shapes, e.g. the output of quantize_static.

    Returns:
        A new, topologically sorted model.
    """
    graph = model.graph
    if any(n.name.startswith(_CPU_FALLBACK_PREFIX) for n in graph.node):
        raise ValueError("keep_fp32_nodes_on_cpu() was already applied to this model.")
    if len(graph.input) != 1:
        raise ValueError(f"Expected a single graph input, got {len(graph.input)}.")

    seen = set()
    for k, node in enumerate(graph.node):  # nodes are matched by name below
        if not node.name or node.name in seen:
            node.name = f"{node.op_type}_{k}"
        seen.add(node.name)

    graph_input = graph.input[0].name
    graph_outputs = {o.name for o in graph.output}
    producer = {o: n for n in graph.node for o in n.output}
    consumers = defaultdict(list)
    for node in graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    initializers = {i.name: i for i in graph.initializer}
    constants = set(initializers)
    for (
        node
    ) in (
        graph.node
    ):  # tensors computed from initializers only, e.g. weight DequantizeLinear
        if all(not i or i in constants for i in node.input):
            constants.update(node.output)

    def is_qdq(node):
        return node.op_type in ("QuantizeLinear", "DequantizeLinear")

    def activation_inputs(node):
        return [i for i in node.input if i and i not in constants]

    def is_quantized(node):
        inputs = activation_inputs(node)
        return (
            bool(inputs)
            and all(
                i in producer and producer[i].op_type == "DequantizeLinear"
                for i in inputs
            )
            and all(
                o not in graph_outputs
                and consumers[o]
                and all(c.op_type == "QuantizeLinear" for c in consumers[o])
                for o in node.output
            )
        )

    fp32_nodes = [
        n
        for n in graph.node
        if not is_qdq(n) and activation_inputs(n) and not is_quantized(n)
    ]
    fp32_names = {n.name for n in fp32_nodes}

    # Cheap quantized nodes followed only by unquantized ones (e.g. flattening and
    # concatenating detection head outputs) are moved to the CPU too.
    def downstream(node):
        """Non-QDQ consumers of node, looking through QuantizeLinear/DequantizeLinear."""
        result, stack = [], list(node.output)
        while stack:
            for c in consumers[stack.pop()]:
                if is_qdq(c):
                    stack.extend(c.output)
                else:
                    result.append(c)
        return result

    tail_names = set()
    for node in reversed(graph.node):  # consumers are visited before producers
        if (
            node.op_type in _CHEAP_TAIL_OPS
            and node.name not in fp32_names
            and not is_qdq(node)
            and all(
                c.name in fp32_names or c.name in tail_names for c in downstream(node)
            )
        ):
            tail_names.add(node.name)
    cpu_names = fp32_names | tail_names

    # Tensors entering a CPU part: the graph input and dequantized activations that come
    # from an accelerated part.
    def from_accelerated_part(dq):
        quantize = producer.get(dq.input[0])
        source = producer.get(quantize.input[0]) if quantize is not None else None
        return source is not None and source.name not in cpu_names

    entries = [
        t
        for t in [graph_input]
        + [
            n.output[0]
            for n in graph.node
            if n.op_type == "DequantizeLinear"
            and n.input[0] not in constants
            and from_accelerated_part(n)
        ]
        if any(c.name in cpu_names for c in consumers[t])
    ]
    # Tensors leaving an unquantized part, i.e. quantized again.
    exits = [
        n.input[0]
        for n in graph.node
        if n.op_type == "QuantizeLinear"
        and n.input[0] in producer
        and producer[n.input[0]].name in fp32_names
    ]

    inferred = shape_inference.infer_shapes(model)
    shapes = {
        vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
        for vi in list(inferred.graph.value_info) + list(inferred.graph.input)
    }

    def is_static(tensor):
        return bool(shapes.get(tensor)) and all(d > 0 for d in shapes[tensor])

    entries = [
        t for t in entries if is_static(t)
    ]  # dynamic shapes already run on the CPU
    for tensor in exits:
        if not is_static(tensor):
            raise ValueError(
                f"Tensor {tensor} needs a fully static shape, got {shapes.get(tensor)}."
            )

    def name(suffix):
        return f"{_CPU_FALLBACK_PREFIX}_{suffix}"

    def const(suffix, values, dtype=np.int64):
        graph.initializer.append(
            numpy_helper.from_array(np.array(values, dtype=dtype), name(suffix))
        )
        return name(suffix)

    big = const("big", 1e30, dtype=np.float32)
    one = const("one", [1])
    new_nodes, rewires = [], []

    def hide_shape(tensor, tag, zero=None):
        # Reshape(tensor, shape + zero), where zero = [int(tensor[0, 0, ...] > 1e30)] is always
        # 0 but computed from the data. The batch dim is 0, i.e. copied from the input.
        # An existing zero computed upstream of tensor can be reused.
        rank = len(shapes[tensor])
        reshape = [
            helper.make_node(
                "Add",
                [
                    const(f"{tag}_shape", [0] + shapes[tensor][1:]),
                    zero or name(f"{tag}_zero"),
                ],
                [name(f"{tag}_shape_dyn")],
                name=name(f"{tag}_AddShape"),
            ),
            helper.make_node(
                "Reshape",
                [tensor, name(f"{tag}_shape_dyn")],
                [name(f"{tag}_out")],
                name=name(f"{tag}_Reshape"),
            ),
        ]
        if zero:
            return reshape
        # For a dequantized tensor, zero is computed from the quantized values (> the type's
        # max, also always false), so no DequantizeLinear is moved onto the helper nodes.
        source, threshold = tensor, big
        dequantize = producer.get(tensor)
        if (
            dequantize is not None
            and dequantize.op_type == "DequantizeLinear"
            and dequantize.input[0] not in constants
            and len(dequantize.input) > 2
            and dequantize.input[2] in initializers
        ):
            dtype = numpy_helper.to_array(initializers[dequantize.input[2]]).dtype
            source = dequantize.input[0]
            threshold = const(f"{tag}_max", np.iinfo(dtype).max, dtype=dtype)
        return [
            helper.make_node(
                "Slice",
                [
                    source,
                    const(f"{tag}_starts", [0] * rank),
                    const(f"{tag}_ends", [1] * rank),
                ],
                [name(f"{tag}_first")],
                name=name(f"{tag}_Slice"),
            ),
            helper.make_node(
                "Greater",
                [name(f"{tag}_first"), threshold],
                [name(f"{tag}_gt")],
                name=name(f"{tag}_Greater"),
            ),
            helper.make_node(
                "Cast",
                [name(f"{tag}_gt")],
                [name(f"{tag}_zero_nd")],
                to=onnx.TensorProto.INT64,
                name=name(f"{tag}_Cast"),
            ),
            helper.make_node(
                "Reshape",
                [name(f"{tag}_zero_nd"), one],
                [name(f"{tag}_zero")],
                name=name(f"{tag}_Zero"),
            ),
        ] + reshape

    for k, tensor in enumerate(entries):
        new_nodes += hide_shape(tensor, f"in{k}")
        rewires.append((tensor, name(f"in{k}_out"), cpu_names))

    for k, tensor in enumerate(exits):
        tag = f"out{k}"
        new_nodes.append(
            helper.make_node(
                "Reshape",
                [tensor, const(f"{tag}_shape", shapes[tensor])],
                [name(f"{tag}_out")],
                name=name(f"{tag}_Reshape"),
            )
        )
        quantize_nodes = {
            c.name for c in consumers[tensor] if c.op_type == "QuantizeLinear"
        }
        rewires.append((tensor, name(f"{tag}_out"), quantize_nodes))

    def apply(new_nodes, rewires):
        for old, new, targets in rewires:
            for node in graph.node:
                if node.name in targets:
                    for i, input_name in enumerate(node.input):
                        if input_name == old:
                            node.input[i] = new
        graph.node.extend(new_nodes)
        sort_nodes_topologically(model)
        del graph.value_info[:]

    apply(new_nodes, rewires)

    # Inside a CPU part, shapes can become static again, e.g. after a Reshape to a constant
    # shape or when broadcasting with a constant. Hide those too, until none are left.
    for round_ in range(100):
        inferred = shape_inference.infer_shapes(model)
        infos = list(inferred.graph.value_info) + list(inferred.graph.input)
        shapes = {
            vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
            for vi in infos
        }
        floats = {
            vi.name
            for vi in infos
            if vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        }
        consumers = defaultdict(list)
        for node in graph.node:
            for input_name in node.input:
                consumers[input_name].append(node)

        def cpu_targets(tensor):
            """CPU consumers of tensor, and QuantizeLinear nodes that only lead to CPU nodes."""
            return {
                c.name
                for c in consumers[tensor]
                if c.name in cpu_names
                or (
                    c.op_type == "QuantizeLinear"
                    and downstream(c)
                    and all(d.name in cpu_names for d in downstream(c))
                )
            }

        static = [
            o
            for node in graph.node
            if node.name in cpu_names
            for o in node.output
            if o in floats
            and o not in graph_outputs
            and len(shapes[o]) > 0
            and is_static(o)
            and cpu_targets(o)
        ]
        if not static:
            break
        producer = {o: n for n in graph.node for o in n.output}
        entry_zeros = {
            name(f"in{k}_out"): name(f"in{k}_zero") for k in range(len(entries))
        }

        def upstream_zero(tensor):
            """The zero of an entry that tensor is computed from, if any."""
            seen, stack = set(), [tensor]
            while stack:
                t = stack.pop()
                if t in entry_zeros:
                    return entry_zeros[t]
                if t in seen or t not in producer:
                    continue
                seen.add(t)
                stack.extend(producer[t].input)
            return None

        def constant_values(tensor):
            if tensor in initializers:
                return numpy_helper.to_array(initializers[tensor])
            node = producer.get(tensor)
            if node is not None and node.op_type == "Constant":
                return numpy_helper.to_array(node.attribute[0].t)
            return None

        new_nodes, rewires = [], []
        for k, tensor in enumerate(static):
            tag = f"cpu{round_}_{k}"
            zero = upstream_zero(tensor)
            targets = [n for n in graph.node if n.name in cpu_targets(tensor)]
            if (
                zero
                and targets
                and all(
                    n.op_type == "Reshape"
                    and n.input[0] == tensor
                    and constant_values(n.input[1]) is not None
                    for n in targets
                )
            ):
                # A Reshape directly after the hiding Reshape would be merged with it by
                # graph optimizers, so the target shape of the existing Reshape is made
                # data-dependent instead.
                for j, reshape in enumerate(targets):
                    shape = const(f"{tag}_{j}_shape", constant_values(reshape.input[1]))
                    new_nodes.append(
                        helper.make_node(
                            "Add",
                            [shape, zero],
                            [name(f"{tag}_{j}_shape_dyn")],
                            name=name(f"{tag}_{j}_AddShape"),
                        )
                    )
                    reshape.input[1] = name(f"{tag}_{j}_shape_dyn")
                continue
            new_nodes += hide_shape(tensor, tag, zero)
            rewires.append((tensor, name(f"{tag}_out"), cpu_targets(tensor)))
        apply(new_nodes, rewires)

    return shape_inference.infer_shapes(model)


def cap_channel_imbalance(
    model: onnx.ModelProto, max_ratio: float = 3.0, compensate: bool = False
):
    """Caps the per-output-channel imbalance of every BN-folded conv, in place.

    Weights are quantized per tensor, so one large channel coarsens the scale of all.

    Args:
        model: Loaded float32 ONNX model, modified in place.
        max_ratio: Bound on max(channel absmax) / median(channel absmax), >= 1.
        compensate: Undo the scaling in the following convs where the graph allows it;
            otherwise the model's outputs change.

    Returns:
        ``(tensors_changed, channels_rescaled)``.
    """
    by_name = {init.name: init for init in model.graph.initializer}
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    convs_by_weight = {
        node.input[1]: node for node in model.graph.node if node.op_type == "Conv"
    }

    def compensation_targets(weight_name):
        conv = convs_by_weight.get(weight_name)
        if conv is None:
            return []
        users = consumers.get(conv.output[0], [])
        if len(users) != 1 or users[0].op_type != "Relu":
            return []
        targets = consumers.get(users[0].output[0], [])
        regular = all(
            t.op_type == "Conv"
            and t.input[0] == users[0].output[0]
            and t.input[1] in by_name
            and next((a.i for a in t.attribute if a.name == "group"), 1) == 1
            for t in targets
        )
        return targets if targets and regular else []

    tensors_seen, tensors_changed, channels_rescaled = 0, 0, 0

    # in node order, so a conv's own cap sees the compensation from its predecessor
    weight_names = [
        node.input[1] for node in model.graph.node if node.op_type == "Conv"
    ]
    for weight_name in weight_names:
        weight_init = by_name.get(weight_name)
        bias_init = by_name.get(weight_name + "_bias")
        if (
            weight_init is None
            or not weight_name.endswith(".conv.weight")
            or bias_init is None
        ):
            continue
        weights = numpy_helper.to_array(weight_init)
        if weights.dtype != np.float32 or weights.ndim != 4:
            continue
        tensors_seen += 1

        absmax = np.abs(weights).max(axis=(1, 2, 3))
        bound = float(np.median(absmax)) * max_ratio
        if bound <= 0 or absmax.max() <= bound:
            continue

        excess = np.maximum(absmax / bound, 1.0).astype(np.float32)
        weight_init.CopyFrom(
            numpy_helper.from_array(
                (weights / excess[:, None, None, None]).astype(np.float32),
                weight_init.name,
            )
        )
        bias_init.CopyFrom(
            numpy_helper.from_array(
                (numpy_helper.to_array(bias_init) / excess).astype(np.float32),
                bias_init.name,
            )
        )
        if compensate:
            for target in compensation_targets(weight_name):
                target_init = by_name[target.input[1]]
                target_weights = numpy_helper.to_array(target_init)
                target_init.CopyFrom(
                    numpy_helper.from_array(
                        (target_weights * excess[None, :, None, None]).astype(
                            np.float32
                        ),
                        target_init.name,
                    )
                )
        tensors_changed += 1
        channels_rescaled += int((excess > 1.0).sum())

    if not tensors_seen:
        # Expected for models without mmcv ConvModule layers (e.g. a torchvision ResNet
        # backbone) and for untrained models, whose all-zero folded biases the exporter
        # drops. Otherwise it means the exporter's naming has changed.
        logger.warning(
            "cap_channel_imbalance found no BN-folded ConvModule conv (*.conv.weight with a "
            "*.conv.weight_bias) to check"
        )
    return tensors_changed, channels_rescaled


def find_final_convs(onnx_model_path):
    """Names of the last Conv nodes of a model, i.e. Convs with no other Conv downstream.

    Args:
        onnx_model_path: Path to the float32 ONNX model.

    Returns:
        List of node names to pass to quantize_static(nodes_to_exclude=...).
    """
    nodes = list(onnx.load(str(onnx_model_path)).graph.node)
    consumers = defaultdict(list)
    for k, node in enumerate(nodes):
        for input_name in node.input:
            consumers[input_name].append(k)

    # ONNX nodes are topologically sorted, so consumers are visited before producers.
    reaches_conv = [False] * len(nodes)
    for k in reversed(range(len(nodes))):
        reaches_conv[k] = any(
            nodes[c].op_type == "Conv" or reaches_conv[c]
            for output_name in nodes[k].output
            for c in consumers[output_name]
        )

    return [
        n.name
        for k, n in enumerate(nodes)
        if n.op_type == "Conv" and not reaches_conv[k]
    ]
