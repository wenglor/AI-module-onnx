import colorsys
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
    BoxesCoordinate,
    BoxesFormat,
    ChannelOrder,
    ClassDict,
    DatasetColorMode,
    InferenceDevice,
    InputColorSpace,
    OutputType,
    Quantization,
    ResizeImageAlignmentHorizontal,
    ResizeImageAlignmentVertical,
    ResizeMode,
)
from utils.image import write_image_file


def validate_classification_onnx_model(onnx_model_path, channel_order, classes):
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


def validate_object_detection_onnx_model(
    onnx_model_path,
    channel_order,
    boxes_output_index,
    labels_output_index,
    scores_output_index,
):
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

    num_outputs = len(onnx_model.graph.output)
    required_indices = [boxes_output_index, labels_output_index, scores_output_index]
    max_index = max(required_indices)

    if num_outputs < 3:
        raise ValueError(
            f"The ONNX model should have at least 3 outputs for object detection, but has {num_outputs}."
        )

    if max_index >= num_outputs:
        raise ValueError(
            f"Output index {max_index} is out of bounds. Model has {num_outputs} outputs (valid indices: 0-{num_outputs-1})."
        )

    boxes_output = onnx_model.graph.output[boxes_output_index]
    boxes_tensor_type = boxes_output.type.tensor_type
    if boxes_tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError(
            f"Boxes output (index {boxes_output_index}) type is not float32."
        )

    boxes_shape_dims = boxes_tensor_type.shape.dim
    if len(boxes_shape_dims) != 2:
        raise ValueError(
            f"Boxes output (index {boxes_output_index}) must be 2D (detections_count, 4), but has {len(boxes_shape_dims)} dimensions."
        )

    second_dim = boxes_shape_dims[1]
    if second_dim.dim_value != 4:
        raise ValueError(
            f"Boxes output (index {boxes_output_index}) second dimension must be 4, but got {second_dim.dim_value}."
        )

    labels_output = onnx_model.graph.output[labels_output_index]
    labels_tensor_type = labels_output.type.tensor_type
    if labels_tensor_type.elem_type != onnx.TensorProto.INT64:
        raise ValueError(
            f"Labels output (index {labels_output_index}) type is not int64."
        )

    labels_shape_dims = labels_tensor_type.shape.dim
    if len(labels_shape_dims) != 1:
        raise ValueError(
            f"Labels output (index {labels_output_index}) must be 1D (detections_count,), but has {len(labels_shape_dims)} dimensions."
        )

    scores_output = onnx_model.graph.output[scores_output_index]
    scores_tensor_type = scores_output.type.tensor_type
    if scores_tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError(
            f"Scores output (index {scores_output_index}) type is not float32."
        )

    scores_shape_dims = scores_tensor_type.shape.dim
    if len(scores_shape_dims) != 1:
        raise ValueError(
            f"Scores output (index {scores_output_index}) must be 1D (detections_count,), but has {len(scores_shape_dims)} dimensions."
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


def export_univision_model_v3(
    univision_model_path: str,
    onnx_model_path: str,
    classes: Union[list[str], list[ClassDict]],
    input_example: np.ndarray,
    model_name: Optional[str] = None,
    model_uuid: Optional[str] = None,
    inference_device: Optional[Union[InferenceDevice, str]] = InferenceDevice.AUTO,
    quantization: Optional[Union[Quantization, str]] = None,
    resize_mode: Optional[Union[ResizeMode, str]] = ResizeMode.STRETCH,
    resize_padding_value: Optional[Union[tuple[int], tuple[int, int, int]]] = None,
    resize_image_alignment_horizontal: Optional[
        Union[ResizeImageAlignmentHorizontal, str]
    ] = None,
    resize_image_alignment_vertical: Optional[
        Union[ResizeImageAlignmentVertical, str]
    ] = None,
    unit_scaling: bool = False,
    standardization_std: Optional[Union[tuple[int], tuple[int, int, int]]] = None,
    standardization_mean: Optional[Union[tuple[int], tuple[int, int, int]]] = None,
    channel_order: Union[ChannelOrder, str] = ChannelOrder.NCHW,
    dataset_color_mode: Union[DatasetColorMode, str] = DatasetColorMode.COLOR,
    input_color_space: Union[InputColorSpace, str] = InputColorSpace.RGB,
    output_type: Union[OutputType, str] = OutputType.OBJECT_DETECTION,
    heatmap_feature_layer: Optional[str] = None,
    class_thresholds: Optional[list[float]] = None,
    class_colors: Optional[list[tuple[int, int, int]]] = None,
    boxes_output_index: Optional[int] = None,
    labels_output_index: Optional[int] = None,
    scores_output_index: Optional[int] = None,
    boxes_format: Optional[Union[BoxesFormat, str]] = None,
    boxes_coordinates: Optional[Union[BoxesCoordinate, str]] = None,
    max_detections: Optional[int] = None,
    zip_password: Optional[str] = None,
):
    """
    Exports a model to a uniVision format, along with metadata and an input example.

    Input preprocessing order:
        1.(optional) Color space/channel transformation
        As a result of this transformation the image data has shape
            a.(batch_size, original_height, original_width, channels) for channel_order=NHWC
            b. (batch_size, channels, original_height, original_width) for channel_order=NCHW
            where channels is 1 or 3 and the image channel data matches the order defined by color_space (RGB/BGR/GRAYSCALE).
        2. Image resize
        3. (optional) Unit scaling
            x = x / 255
        4.(optional) Standardization
            x = (x - standardization_mean) / standardization_std

    Args:
        univision_model_path (str): Path where the uniVision model will be saved.
        onnx_model_path (str): Path to the ONNX model file.
        classes (Union[list[str], list[ClassDict]]): Class labels.
            If provided as a list of strings, it will be automatically converted to a list of dictionaries.
        input_example (np.ndarray): Example input image or data to validate the model.
        model_name (Optional[str]): An optional name for the model, which can be used as a human-friendly identifier.
            If the model_name field is missing or null, it is treated as an empty string.
        model_uuid (Optional[str]): An optional model UUID. Automatically generated if not provided.
        inference_device (Optional[Union[InferenceDevice, str]]): The device to use for inference. Either "AUTO" or "CPU_ONLY".
        quantization (Optional[Union[Quantization, str]]): Model quantization settings.
            If None, no quantization is applied. Defaults to None.
        resize_mode (Optional[Union[ResizeMode, str]]): Options are "STRETCH" which ignores aspect ratio and "FIT_WITH_PADDING" which
            maintains aspect ratio.
        resize_padding_value (Optional[Union[tuple[int], tuple[int, int, int]]]): Only valid if mode is `FIT_WITH_PADDING`. Defines the constant color used for
            padding. Defaults to (0, 0, 0) or (0).
        resize_image_alignment_horizontal (Optional[Union[ResizeImageAlignmentHorizontal, str]])): Only valid if mode is `FIT_WITH_PADDING`.
            Defines horizontal alignment of the resized image. Either "LEFT", "CENTER" or "RIGHT". Defaults to "CENTER".
        resize_image_alignment_vertical (Optional[Union[ResizeImageAlignmentVertical, str]])): nly valid if mode is `FIT_WITH_PADDING`.
            Defines vertical alignment of the resized image. Either "TOP", "CENTER" or "BOTTOM". Defaults to "CENTER".
        unit_scaling (bool): Whether to apply unit scaling. Defaults to True.
        standardization_std (Optional[Union[tuple[int], tuple[int, int, int]]]): Standard deviation values for input
            standardization. Must be a tuple of floats or None. Defaults to None.
        standardization_mean (Optional[Union[tuple[int], tuple[int, int, int]]]): Mean values for input standardization.
            Must be a tuple of floats or None. Defaults to None.
        channel_order (Union[ChannelOrder, str]): Specifies the channel order of the input.
            Can be either 'NHWC' or 'NCHW'. Defaults to 'NCHW'.
        dataset_color_mode (Union[DatasetColorMode, str]): Color mode of the dataset.
            Either 'COLOR' or 'MONOCHROME'. Defaults to 'COLOR'.
        input_color_space (Union[InputColorSpace, str]): Color space of the input.
            Either 'RGB' or 'BGR'. Defaults to 'RGB'.
        output_type (Union[OutputType, str]): Output type of the model.
            E.g., 'OBJECT_DETECTION'. Defaults to 'OBJECT_DETECTION'.
        heatmap_feature_layer (Optional[str]): Heatmap feature layer name of the model.
            Defaults to None.
        class_thresholds (Optional[list[float]]): Only applicable to multi label and object detection. Threshold values of classes.
            The range of allowable values is [0,1] and must be provided for all the output classes.
            If omitted it will be used 0.5 for all the output classes.
        class_colors (Optional[list[tuple[int, int, int]]]): Only applicable to object detection. RGB values for bounding box
            visualization. If omitted, they will be generated based on the number of classes.
        boxes_output_index (Optional[int]): Only applicable to object detection. The index of the ONNX output with bounding boxes
            per detection. Defaults to 0.
        labels_output_index (Optional[int]): Only applicable to object detection. The index of the ONNX output with predicted labels
            per detection. Defaults to 1.
        scores_output_index (Optional[int]): Only applicable to object detection. The index of the ONNX output with confidence scores
            per detection. Defaults to 2.
        boxes_format (Optional[BoxesFormat, str]):  Only applicable to object detection. Describes the format of bounding boxes in the corresponding ONNX output
            with index boxes_output_index. Either 'center_size' or 'left_top_right_bottom' or 'top_left_size'.
            Defaults to 'left_top_right_bottom'.
        boxes_coordinates (Optional[BoxesCoordinate, str]) : Only applicable to object detection. Describes the format of bounding boxes coordinates in the
            corresponding ONNX output with index boxes_output_index. Either 'relative' or 'absolute'. Defaults to 'absolute'.
        max_detections (int): Only applicable to object detection. uniVision uses this to filter the detections of the ONNX model
            if it returns too many. Defaults to 20.

    Returns:
        None: This function saves a zipped Univision model file at the specified path.
    """

    validate_enum("channel_order", channel_order, ChannelOrder)
    validate_enum("dataset_color_mode", dataset_color_mode, DatasetColorMode)
    validate_enum("input_color_space", input_color_space, InputColorSpace)
    validate_enum("output_type", output_type, OutputType)
    validate_enum("quantization", quantization, Optional[Quantization])
    validate_enum("inference_device", inference_device, InferenceDevice)
    validate_enum("resize_mode", resize_mode, ResizeMode)
    validate_enum(
        "resize_image_alignment_horizontal",
        resize_image_alignment_horizontal,
        Optional[ResizeImageAlignmentHorizontal],
    )
    validate_enum(
        "resize_image_alignment_vertical",
        resize_image_alignment_vertical,
        Optional[ResizeImageAlignmentVertical],
    )

    # validate object detection args and validate onnx model
    outputs_extra_args = {}
    if output_type != OutputType.OBJECT_DETECTION and (
        boxes_output_index is not None
        or labels_output_index is not None
        or scores_output_index is not None
        or max_detections is not None
        or boxes_format is not None
        or boxes_coordinates is not None
        or class_colors is not None
    ):
        raise ValueError(
            "boxes_output_index, labels_output_index, scores_output_index, max_detections, boxes_format, boxes_coordinates and class_colors are only supported for object detection."
        )
    elif output_type == OutputType.OBJECT_DETECTION:

        boxes_output_index = 0 if boxes_output_index is None else boxes_output_index
        labels_output_index = 1 if labels_output_index is None else labels_output_index
        scores_output_index = 2 if scores_output_index is None else scores_output_index

        max_detections = 20 if max_detections is None else max_detections

        validate_enum("boxes_format", boxes_format, BoxesFormat)
        validate_enum("boxes_coordinates", boxes_coordinates, BoxesCoordinate)

        index_check = {
            "boxes_output_index": boxes_output_index,
            "labels_output_index": labels_output_index,
            "scores_output_index": scores_output_index,
        }
        if len(set(index_check.values())) != len(index_check):
            raise ValueError(
                "Values for boxes_output_index, labels_output_index and scores_output_index must be unique."
            )

        non_zero_integer_check = index_check
        non_zero_integer_check.update({"max_detections": max_detections})
        for variable_name, variable in non_zero_integer_check.items():
            if not isinstance(variable, int):
                raise ValueError(f"{variable_name} must be an integer.")
            if variable < 0:
                raise ValueError(f"{variable_name} must be positive.")

        input_width, input_height, input_channels = (
            validate_object_detection_onnx_model(
                onnx_model_path,
                channel_order,
                boxes_output_index,
                labels_output_index,
                scores_output_index,
            )
        )

        if class_colors is None:
            class_colors = generate_distinct_colors(len(classes))
        else:
            if len(class_colors) != len(classes):
                raise ValueError(
                    f"The number of class colors specified ({len(class_colors)}) does not match the number of model outputs ({len(classes)})."
                )
            for class_color in class_colors:
                for channel in class_color:
                    if not isinstance(channel, int) or channel < 0 or channel > 255:
                        raise ValueError(
                            f"Color values must be integers in the range [0, 255]. (wrong value/s: {class_color})"
                        )

        outputs_extra_args.update(
            {
                "boxes_output_index": boxes_output_index,
                "labels_output_index": labels_output_index,
                "scores_output_index": scores_output_index,
                "boxes_format": str(boxes_format),
                "boxes_coordinates": str(boxes_coordinates),
                "max_detections": max_detections,
            }
        )
    elif output_type in [
        OutputType.MULTI_LABEL_CLASSIFICATION,
        OutputType.MULTI_CLASS_CLASSIFICATION,
    ]:
        input_width, input_height, input_channels = validate_classification_onnx_model(
            onnx_model_path, channel_order, classes
        )

    # validate resize
    resize_additional_options = {}
    if resize_mode == ResizeMode.FIT_WITH_PADDING:
        if resize_image_alignment_horizontal is None:
            resize_image_alignment_horizontal = ResizeImageAlignmentHorizontal.CENTER
        if resize_image_alignment_vertical is None:
            resize_image_alignment_vertical = ResizeImageAlignmentVertical.CENTER
        if resize_padding_value is None:
            resize_padding_value = (0, 0, 0) if input_channels == 3 else (0)

        if not isinstance(resize_padding_value, tuple):
            raise ValueError("Resize padding value must be a tuple.")

        if len(resize_padding_value) != input_channels:
            raise ValueError(
                f"length of resize_padding_value ({len(resize_padding_value)}) is not equal to input_channels ({input_channels})"
            )

        for value in resize_padding_value:
            if not isinstance(value, int) or value < 0 or value > 255:
                raise ValueError(
                    "Resize padding value must be an integer between 0 and 255."
                )

        resize_additional_options.update(
            {
                "padding_value": resize_padding_value,
                "image_alignment": {
                    "horizontal": str(resize_image_alignment_horizontal),
                    "vertical": str(resize_image_alignment_vertical),
                },
            }
        )
    elif resize_mode == ResizeMode.STRETCH:
        if (
            resize_image_alignment_horizontal is not None
            or resize_image_alignment_vertical is not None
        ):
            raise ValueError(
                "Resize alignment is not supported for stretch resize mode."
            )
        if resize_padding_value is not None:
            raise ValueError(
                "Resize padding value is not supported for stretch resize mode."
            )

    # validate heatmap_feature_layer
    if output_type in [
        OutputType.MULTI_LABEL_CLASSIFICATION,
        OutputType.MULTI_CLASS_CLASSIFICATION,
    ]:
        if heatmap_feature_layer is not None:
            if not isinstance(heatmap_feature_layer, str):
                raise ValueError("Heatmap feature layer must be a string.")
    else:
        if heatmap_feature_layer is not None:
            raise ValueError(
                "Heatmap feature layer is only available for multi label and multi class classification."
            )

    # validate unit scaling
    if not isinstance(unit_scaling, bool):
        raise ValueError(f"Unit_scaling options are [True, False].")

    # validate normalization
    tuple_of_floats_check = {
        "standardization_std": standardization_std,
        "standardization_mean": standardization_mean,
    }
    for variable_name, variable in tuple_of_floats_check.items():
        if variable:
            if isinstance(variable, tuple):
                if not all([isinstance(x, float) for x in variable]):
                    raise ValueError(
                        f"{variable_name} must be a tuple of floats or None."
                    )
                elif len(variable) != input_channels:
                    raise ValueError(
                        f"The length of {variable_name} must be the equal to `input_channels`."
                    )
            else:
                raise ValueError(f"{variable_name} must be a tuple of floats or None.")

    # validate dataset color mode
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

    # class threshold check
    if (
        output_type == OutputType.MULTI_CLASS_CLASSIFICATION
        and class_thresholds is not None
    ):
        raise ValueError(
            f"Multi Class output was selected but class threshold values were provided."
        )

    if output_type in [
        OutputType.MULTI_LABEL_CLASSIFICATION,
        OutputType.OBJECT_DETECTION,
    ]:
        if class_thresholds is None:
            class_thresholds = [0.5] * len(classes)
        else:
            if len(class_thresholds) != len(classes):
                raise ValueError(
                    f"The number of class thresholds specified ({len(class_thresholds)}) does not match the number of model outputs ({len(classes)})."
                )

            invalid_indexes = [
                i
                for i, item in enumerate(class_thresholds)
                if not isinstance(item, float) or item < 0.0 or item > 1.0
            ]
            if invalid_indexes:
                raise ValueError(
                    f"Threshold values of classes must be of type float and [0,1]. (wrong value/s: {[class_thresholds[idx] for idx in invalid_indexes]})"
                )

    # validate model_uuid
    if model_uuid is None:
        model_uuid = str(uuid.uuid4())
    else:
        uuid.UUID(model_uuid, version=4)

    # validate classes
    classes = validate_classes(classes)

    # validate univision model path
    if Path(univision_model_path).suffix != ".u3o":
        raise ValueError(
            f"File '{univision_model_path}' does not have a '.u3o' extension."
        )

    # create metadata
    metadata = {
        "metadata_version": SingleQuotedScalarString("3.0.0"),
        "model_uuid": SingleQuotedScalarString(model_uuid),
        "model_name": SingleQuotedScalarString(model_name),
        "creation_time": SingleQuotedScalarString(
            datetime.datetime.utcnow().isoformat(timespec="seconds")
        ),
        "quantization": str(quantization),
        "inference_device": str(inference_device),
        "dataset_color_mode": str(dataset_color_mode),
        **(
            {"heatmap_feature_layer": SingleQuotedScalarString(heatmap_feature_layer)}
            if heatmap_feature_layer is not None
            else {}
        ),
        "input": {
            "width": input_width,
            "height": input_height,
            "channels": input_channels,
            "channel_order": str(channel_order),
            "color_space": str(input_color_space),
            "resize": {"mode": resize_mode, **resize_additional_options},
            "unit_scaling": unit_scaling,
            "standardization_std": standardization_std,
            "standardization_mean": standardization_mean,
        },
        "outputs": [
            {
                "type": str(output_type),
                **outputs_extra_args,
                "classes": [
                    {
                        "uuid": SingleQuotedScalarString(c["uuid"]),
                        "name": SingleQuotedScalarString(c["name"]),
                        **(
                            {"color": class_colors[idx]}
                            if class_colors is not None
                            else {}
                        ),
                        **(
                            {"default_threshold": class_thresholds[idx]}
                            if class_thresholds is not None
                            else {}
                        ),
                    }
                    for idx, c in enumerate(classes)
                ],
            }
        ],
    }
    metadata["input"] = {k: v for k, v in metadata["input"].items() if v is not None}
    metadata["outputs"] = [
        {k: v for k, v in output.items() if v is not None}
        for output in metadata["outputs"]
    ]
    if model_name is None:
        metadata.pop("model_name")
    if quantization is None:
        metadata.pop("quantization")

    # create zip
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

    print(f"Successfully exported to {univision_model_path}")


def generate_distinct_colors(
    n: int, saturation: float = 1.0, value: float = 1.0
) -> list[list[int, int, int]]:
    """
    Generate n visually distinct RGB colors suitable for bounding box visualization.

    The colors are generated in HSV space with evenly distributed hues using the golden ratio
    for better perceptual separation, even for large n.

    Args:
        n (int): Number of colors to generate (n >= 0).
        saturation (float): Saturation value in HSV (0.0 to 1.0, default 1.0 for vivid colors).
        value (float): Value/brightness in HSV (0.0 to 1.0, default 1.0 for bright colors).

    Returns:
        list[list[int, int, int]]: List of n RGB lists, each with values 0-255.
    """
    if n <= 0:
        return []

    colors = []
    golden_ratio_conjugate = 0.618033988749895
    hue = 0.0

    for _ in range(n):
        hue = (hue + golden_ratio_conjugate) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        # Round to nearest integer and convert to 0-255 range
        colors.append([int(round(r * 255)), int(round(g * 255)), int(round(b * 255))])

    return colors
