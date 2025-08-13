#### 4.1.2 `.yaml` metadata file description

| Name | Type | Required | Description |
|:-:|:-:|:-:|:-|
| **Root node** | | | |
| metadata_version | str | yes | The version in SemVer format. |
| model_uuid | str | no | A unique model identifier to find it in the logs of a training service. Must be a UUID version 4 (randomly generated), RFC 4122. If not provided, a `model_uuid` will be automatically generated. |
| model_name | str | no | A name for the model, which can be used as a human-friendly identifier. If the `model_name` field is missing or null, it is treated as an empty string. |
| creation_time | str | yes | The time when a uniVision ONNX model is exported. Time uses ISO 8601 format without time zones (the time zone is UTC by convention) and without microseconds e.g. '2023-11-01T18:19:30'. |
| quantization | str | no | Quantization of weights (if any):<ul><li>`null` (default, no quantization is applied, weights are in `FLOAT32` format)</li><li>`INT8` (only such models can run on an NPU)</li> |
| heatmap_feature_layer | str | no | The name of the ONNX layer, which output is used to produce a saliency map. For example `/backbone/layer4/layer4.1/Add_quant` or `/backbone/layer4/layer4.1/Add`. |
| dataset_color_mode | str | yes | One of:<ul><li>`MONOCHROME`</li><li>`COLOR`</ul>A property of dataset on which a model was trained, to avoid train/test distribution shift the univision Module Image ONNX must be connected to the data source of the same format.<br>Valid combinations of `dataset_color_mode` and `input:color_space`:<ul><li>`dataset_color_mode=MONOCHROME` and `input:color_space=GRAYSCALE`</li><li>`dataset_color_mode=MONOCHROME` and `input:color_space=RGB/BGR` when data is copied into 3 channels of model input</li><li>`dataset_color_mode=COLOR` and `input:color_space=RGB/BGR`</ul>Invalid combinatrions:<ul><li>`dataset_color_mode=COLOR` and `input:color_space=GRAYSCALE`</li></ul>Export your model with:<ul><li>`COLOR` if a training dataset contains color images</li><li>`MONOCHROME` otherwise</li></ul> |
| **Input section** | | | |
| width | int | yes | Image width. |
| height | int | yes | Image height. |
| channels | int | yes | The number of channels of the ONNX file. |
| unit_scaling | bool | no | Specifies if the pixel intensities has to be rescaled from 0..255 to 0.0..1.0 range by dividing them by 255. By default it is `false`. Unit scaling is performed before standardization. This key can be omitted, its default value must be used in this case. |
| standardization_std | seq[float] | no | Standard deviation values per channel as defined by `channel_order` used to divide input values, the number of values depends on channels. If `null` (default), no division is performed. This key can be omitted, its default value must be used in this case. |
| standardization_mean | seq[float] | no | Mean value per channel as defined by `channel_order` used to subtract from input values, the number of values depends on channels. If `null` (default), no subtraction is performed. Standardization runs after unit scaling. During standardization mean subtraction is performed before division by std. This key can be omitted, its default value must be used in this case. |
| channel_order | str | yes | Format of the model input:<ul><li>`NHWC` - batch size, height, width, channels</li><li>`NCHW` - batch size,  channels, height, width</li></ul>|
| color_space | str | yes | Image color space as expected by a model. uniVision Module Image ONNX expects colored inputs in case of `RGB` / `BGR` (and reorders channels according to the needs of the ONNX model if this is required) and grayscale inputs in case of `GRAYSCALE`:<ul><li>`RGB`</li><li>`BGR`</li><li>`GRAYSCALE`</li></ul>Allowed combinations of channels and `color_space`:<ul><li>`color_space=GRAYSCALE channels=1`</li><li>`color_space=RGB/BGR channels=3`</li></ul>Disallowed combinations of channels and color_space:<ul><li>`color_space=GRAYSCALE channels=3`</li><li>`color_space=RGB/BGR channels=1`</li></ul> |
| **Outputs section** | | | |
| type | str | yes | Two types of models are supported: <ul><li>`MULTI_CLASS_CLASSIFICATION`</li><li>`MULTI_LABEL_CLASSIFICATION`</li></ul>
| classes | seq[str] | yes | The value is a list of strings. The length of the list is the number of classes and it must match the corresponding output dimension. The class names should be listed in the same order as their corresponding indices in the ONNX output. |
| class_threshold | seq[float] | no | The class thresholds for all the model ouputs. These are used to convert the model outputs into predictions. If the output score is higher than the threshold the class is considered predicted.  |