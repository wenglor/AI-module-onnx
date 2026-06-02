import os
from pathlib import Path

import onnx
import onnxruntime
import torch
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
    """Finds the node names of first conv, softmax and last gemm.
    Excluding these nodes is a best practice for minimizing quantization degradation"""

    all_nodes = onnx_model.graph.node
    first_conv_name = next(
        (node.name for node in all_nodes if node.op_type == "Conv"), None
    )
    gemm_nodes = [node.name for node in all_nodes if node.op_type == "Gemm"]
    last_gemm_name = gemm_nodes[-1] if gemm_nodes else None

    nodes_to_exclude = [
        node.name
        for node in onnx_model.graph.node
        if "Softmax" in node.name
        or node.name == first_conv_name
        or node.name == last_gemm_name
    ]
    return nodes_to_exclude


def sort_nodes_topologically(model: onnx.ModelProto):
    """Reorder graph nodes into "latest-possible" topological order.

    QDQ models exported by PyTorch/ORT place weight DequantizeLinear nodes near
    the top of the node list even though they are consumed only by layers deep in
    the network.  The C++ SplitONNXModel partitions by sequential list position,
    so those early weight nodes get stranded in the wrong half.

    This function schedules every node as late as possible: a node is emitted
    only after all nodes that consume its outputs have been emitted (in reverse).
    Concretely it runs a reverse Kahn's DFS from the graph outputs, collecting
    nodes in reverse execution order, then reverses the result.  Weight
    DequantizeLinear nodes therefore land immediately before the Conv/Gemm nodes
    that use them, making any sequential split correct without needing to know
    the split point in advance.

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
