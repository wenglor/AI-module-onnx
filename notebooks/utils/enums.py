from enum import StrEnum
from typing import TypedDict


class ClassDict(TypedDict):
    name: str
    uuid: str


class ChannelOrder(StrEnum):
    NHWC = "NHWC"
    NCHW = "NCHW"


class DatasetColorMode(StrEnum):
    COLOR = "COLOR"
    MONOCHROME = "MONOCHROME"


class InputColorSpace(StrEnum):
    RGB = "RGB"
    BGR = "BGR"
    GRAYSCALE = "GRAYSCALE"


class OutputType(StrEnum):
    BINARY_CLASSIFICATION = "BINARY_CLASSIFICATION"
    MULTI_CLASS_CLASSIFICATION = "MULTI_CLASS_CLASSIFICATION"
    MULTI_LABEL_CLASSIFICATION = "MULTI_LABEL_CLASSIFICATION"
    OBJECT_DETECTION = "OBJECT_DETECTION"


class Quantization(StrEnum):
    INT8 = "INT8"
    FLOAT16 = "FLOAT16"


class BoxesFormat(StrEnum):
    center_size = "center_size"
    left_top_right_bottom = "left_top_right_bottom"
    top_left_size = "top_left_size"


class BoxesCoordinate(StrEnum):
    relative = "relative"
    absolute = "absolute"


class InferenceDevice(StrEnum):
    AUTO = "AUTO"
    CPU_ONLY = "CPU_ONLY"


class ResizeMode(StrEnum):
    STRETCH = "STRETCH"
    FIT_WITH_PADDING = "FIT_WITH_PADDING"


class ResizeImageAlignmentHorizontal(StrEnum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


class ResizeImageAlignmentVertical(StrEnum):
    TOP = "TOP"
    CENTER = "CENTER"
    BOTTOM = "BOTTOM"
