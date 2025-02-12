import datetime
import enum
import shutil
import uuid
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple, TypedDict, Union

import cv2
import numpy as np
import onnx
import pyzipper
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import SingleQuotedScalarString
from utils.enums import (
    ChannelOrder,
    ClassDict,
    DatasetColorMode,
    InputColorSpace,
    OutputType,
    Quantization,
)
from utils.image import write_image_file


def validate_onnx_model(onnx_model_path, channel_order, classes):
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    if len(onnx_model.graph.input) != 1:
        raise ValueError("The number of inputs of ONNX model is not 1.")

    input_tensor_type = onnx_model.graph.input[0].type.tensor_type
    if input_tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("ONNX input type is not float32.")

    onnx_input_shape = [d.dim_value for d in input_tensor_type.shape.dim]

    if len(onnx_input_shape) != 4:
        raise ValueError("ONNX input ndims should be 4.")
    if onnx_input_shape[0] != 1:
        raise ValueError("The batch size should be 1.")

    if channel_order == ChannelOrder.NHWC:
        _, input_height, input_width, input_channels = onnx_input_shape
    elif channel_order == ChannelOrder.NCHW:
        _, input_channels, input_height, input_width = onnx_input_shape

    if input_channels not in [1, 3]:
        raise ValueError("Input_channels options are [1, 3].")

    if len(onnx_model.graph.output) != 1:
        raise ValueError("The ONNX model should have exactly one output.")

    output_tensor_type = onnx_model.graph.output[0].type.tensor_type
    if output_tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("ONNX output type is not float32.")

    onnx_output_shape = [d.dim_value for d in output_tensor_type.shape.dim]

    class_count = len(classes)
    if onnx_output_shape != [1, class_count]:
        raise ValueError(
            f"Output shape should be [1, {class_count}], but got {onnx_output_shape}."
        )

    return input_width, input_height, input_channels


def validate_enum(parameter_name, value, enum_class):
    # check if if enum_class is an instance of Optional[SomeType]:
    if enum_class.__name__ == "Optional":
        if value is None:
            return

        enum_class = enum_class.__args__[0]

    if type(value) is enum_class:
        return

    try:
        parsed_value = enum_class(value)
    except ValueError:
        raise ValueError(
            f"{value} is not valid for {parameter_name} of type {enum_class.__name__} which options are {list(enum_class.__members__.keys())}."
        )


def validate_classes(classes: Union[List[str], List[ClassDict]]) -> List[ClassDict]:

    if all(isinstance(c, str) for c in classes):
        classes = [{"uuid": str(uuid.uuid4()), "name": c} for c in classes]
    elif all(isinstance(c, dict) for c in classes):
        required_keys = {"uuid", "name"}
        for c in classes:
            existing_keys = set(c.keys())
            if not existing_keys == required_keys:
                raise ValueError(
                    f"Invalid keys in dictionary. Expected keys: {required_keys}. Found: {existing_keys}"
                )
            try:
                uuid.UUID(c["uuid"], version=4)
            except:
                raise ValueError(f"Invalid UUID format: {c['uuid']}")
            if not isinstance(c["name"], str):
                raise ValueError(f"'name' must be a string. Found: {c['name']}")
    else:
        raise ValueError(
            "Input must be a list of strings or a list of dictionaries with 'name' and 'uuid' keys."
        )
    return classes


def export_classification_model(
    univision_model_path: str,
    onnx_model_path: str,
    classes: Union[list[str], list[ClassDict]],
    input_example: np.ndarray,
    model_name: Optional[str] = None,
    model_uuid: Optional[str] = None,
    quantization: Optional[Union[Quantization, str]] = None,
    unit_scaling: bool = True,
    standardization_std: Optional[List[float]] = None,
    standardization_mean: Optional[List[float]] = None,
    channel_order: Union[ChannelOrder, str] = ChannelOrder.NCHW,
    dataset_color_mode: Union[DatasetColorMode, str] = DatasetColorMode.COLOR,
    input_color_space: Union[InputColorSpace, str] = InputColorSpace.RGB,
    output_type: Union[OutputType, str] = OutputType.MULTI_CLASS_CLASSIFICATION,
    heatmap_feature_layer: Optional[str] = None,
):
    """
    Exports a classification model to a Univision format, along with metadata and an input example.

    Args:
        univision_model_path (str): Path where the Univision model will be saved.
        onnx_model_path (str): Path to the ONNX model file.
        classes (Union[list[str], list[ClassDict]]): Class labels.
            If provided as a list of strings, it will be automatically converted to a list of dictionaries.
        input_example (np.ndarray): Example input image or data to validate the model.
        model_name (Optional[str]): An optional name for the model, which can be used as a human-friendly identifier.
            If the model_name field is missing or null, it is treated as an empty string.
        model_uuid (Optional[str]): An optional model UUID. Automatically generated if not provided.
        quantization (Optional[Union[Quantization, str]]): Model quantization settings.
            If None, no quantization is applied. Defaults to None.
        unit_scaling (bool): Whether to apply unit scaling. Defaults to True.
        standardization_std (Optional[List[float]]): Standard deviation values for input
            standardization. Must be a list of floats or None. Defaults to None.
        standardization_mean (Optional[List[float]]): Mean values for input standardization.
            Must be a list of floats or None. Defaults to None.
        channel_order (Union[ChannelOrder, str]): Specifies the channel order of the input.
            Can be either 'NHWC' or 'NCHW'. Defaults to 'NCHW'.
        dataset_color_mode (Union[DatasetColorMode, str]): Color mode of the dataset.
            Either 'COLOR' or 'MONOCHROME'. Defaults to 'COLOR'.
        input_color_space (Union[InputColorSpace, str]): Color space of the input.
            Either 'RGB' or 'BGR'. Defaults to 'RGB'.
        output_type (Union[OutputType, str]): Output type of the model.
            E.g., 'MULTI_CLASS_CLASSIFICATION'. Defaults to 'MULTI_CLASS_CLASSIFICATION'.
        heatmap_feature_layer (Optional[str]): Heatmap feature layer name of the model.
            Defaults to None.

    Returns:
        None: This function saves a zipped Univision model file at the specified path.
    """

    validate_enum("channel_order", channel_order, ChannelOrder)
    validate_enum("dataset_color_mode", dataset_color_mode, DatasetColorMode)
    validate_enum("input_color_space", input_color_space, InputColorSpace)
    validate_enum("output_type", output_type, OutputType)
    validate_enum("quantization", quantization, Optional[Quantization])

    input_width, input_height, input_channels = validate_onnx_model(
        onnx_model_path, channel_order, classes
    )

    if not isinstance(unit_scaling, bool):
        raise ValueError(f"Unit_scaling options are [True, False].")

    list_of_floats_check = {
        "standardization_std": standardization_std,
        "standardization_mean": standardization_mean,
    }
    for variable_name, variable in list_of_floats_check.items():
        if variable:
            if isinstance(variable, list):
                if not all([isinstance(x, float) for x in variable]):
                    raise ValueError(
                        f"{variable_name} must be a list of floats or None."
                    )
                elif len(variable) != input_channels:
                    raise ValueError(
                        f"The length of {variable_name} must be the equal to `input_channels`."
                    )
            else:
                raise ValueError(f"{variable_name} must be a list of floats or None.")

    if dataset_color_mode == DatasetColorMode.COLOR and (
        len(input_example.shape) == 2 or input_example.shape[2] != 3
    ):
        raise ValueError(
            f"Input_example shape {input_example.shape} does not match color_space {dataset_color_mode}."
        )
    if dataset_color_mode == DatasetColorMode.MONOCHROME and (
        len(input_example.shape) == 3 and input_example.shape[2] != 1
    ):
        raise ValueError(
            f"Input_example shape {input_example.shape} does not match color_space {dataset_color_mode}."
        )

    if model_uuid is None:
        model_uuid = str(uuid.uuid4())
    else:
        uuid.UUID(model_uuid, version=4)

    classes = validate_classes(classes)

    if Path(univision_model_path).suffix != ".u3o":
        raise ValueError(
            f"File '{univision_model_path}' does not have a '.u3o' extension."
        )

    metadata = {
        "metadata_version": SingleQuotedScalarString("1.0.0"),
        "model_uuid": SingleQuotedScalarString(model_uuid),
        "model_name": SingleQuotedScalarString(model_name),
        "creation_time": SingleQuotedScalarString(
            datetime.datetime.utcnow().isoformat(timespec="seconds")
        ),
        "quantization": str(quantization),
        "heatmap_feature_layer": SingleQuotedScalarString(heatmap_feature_layer),
        "dataset_color_mode": str(dataset_color_mode),
        "input": {
            "width": input_width,
            "height": input_height,
            "channels": input_channels,
            "unit_scaling": unit_scaling,
            "standardization_std": standardization_std,
            "standardization_mean": standardization_mean,
            "channel_order": str(channel_order),
            "color_space": str(input_color_space),
        },
        "outputs": [
            {
                "type": str(output_type),
                "classes": [
                    {
                        "uuid": SingleQuotedScalarString(c["uuid"]),
                        "name": SingleQuotedScalarString(c["name"]),
                    }
                    for c in classes
                ],
            }
        ],
    }
    metadata["input"] = {k: v for k, v in metadata["input"].items() if v}
    metadata["outputs"] = [
        {k: v for k, v in output.items() if v} for output in metadata["outputs"]
    ]
    if model_name is None:
        metadata.pop("model_name")
    if quantization is None:
        metadata.pop("quantization")

    with TemporaryDirectory() as tmp_dir:
        yaml_file_path = tmp_dir / Path("model.yaml")
        input_example_file_path = tmp_dir / Path("input_example.png")
        copied_onnx_model_path = tmp_dir / Path("model.onnx")
        file_paths = [yaml_file_path, input_example_file_path, copied_onnx_model_path]

        shutil.copy(onnx_model_path, copied_onnx_model_path)

        yaml = YAML()
        yaml.version = (1, 2)
        yaml.default_flow_style = False
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.dump(metadata, yaml_file_path)

        write_image_file(input_example, str(input_example_file_path))

        with pyzipper.ZipFile(
            univision_model_path, "w", compression=pyzipper.ZIP_DEFLATED
        ) as zip:
            for file_path in file_paths:
                zip.write(file_path, file_path.name)
                