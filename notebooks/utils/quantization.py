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


def quantize_onnx_model(fp32_onnx_model_path, int8_onnx_model_path, dataset):
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
        quant_format=QuantFormat.QOperator,
        calibration_data_reader=calibration_data_reader,
        activation_type=QuantType.QUInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=output_softmax_node_names,
    )