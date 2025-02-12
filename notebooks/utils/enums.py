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


class Quantization(StrEnum):
    INT8 = "INT8"
    FLOAT16 = "FLOAT16"
