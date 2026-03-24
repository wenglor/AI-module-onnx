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


def quantize_classification_onnx_model(
    fp32_onnx_model_path, int8_onnx_model_path, dataset
):
    """
    Quantizes a floating-point ONNX model to int8 precision using a given dataset for calibration.

    Args:
        fp32_onnx_model_path (str): Path to the floating-point ONNX model.
        int8_onnx_model_path (str): Path to save the quantized int8 ONNX model.
        dataset (torch.utils.data.Dataset): PyTorch dataset used for calibration during quantization.

    Returns:
        None
    """
    fp32_onnx_model_path = Path(fp32_onnx_model_path)
    dir_path = fp32_onnx_model_path.parent
    stem = fp32_onnx_model_path.stem
    suffix = fp32_onnx_model_path.suffix
    preprocessed_fp32_onnx_model_path = dir_path / f"preprocessed_{stem}{suffix}"

    quant_pre_process(
        fp32_onnx_model_path,
        preprocessed_fp32_onnx_model_path,
    )
    onnx_model = onnx.load(preprocessed_fp32_onnx_model_path)
    output_softmax_node_names = [
        node.name for node in onnx_model.graph.node if "Softmax" in node.name
    ]
    calibration_data_reader = TorchCalibrationDataReader(
        preprocessed_fp32_onnx_model_path,
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        samples=500,
    )
    quantize_static(
        preprocessed_fp32_onnx_model_path,
        int8_onnx_model_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=output_softmax_node_names,
        extra_options={
            "OptimizeModel": True,
        },
    )
