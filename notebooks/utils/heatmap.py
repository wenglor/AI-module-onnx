import onnx


def get_heatmap_feature_layer(backbone: str, quantization: bool) -> str:
    """
    Returns the heatmap feature layer name for the given backbone architecture.

    Args:
        backbone (str): The name of the model backbone.
        quantization (bool): A flag indicating whether to return the quantized feature layer.
                             If False, the '_quant' suffix will be removed from the layer name.

    Returns:
        str: The name of the heatmap feature layer for the given backbone.

    Raises:
        ValueError: If the provided backbone is not supported.
    """
    if backbone == "RegNet_X_3_2GF":
        recipro_cam_feature_layer = (
            "/backbone/trunk_output/block4/block4-1/activation/Relu_output_0_QuantizeLinear"
            if quantization
            else "/backbone/trunk_output/block4/block4-1/Add"
        )
    elif backbone == "RegNet_X_1_6GF":
        recipro_cam_feature_layer = (
            "/backbone/trunk_output/block4/block4-1/activation/Relu_output_0_QuantizeLinear"
            if quantization
            else "/backbone/trunk_output/block4/block4-1/Add"
        )
    else:
        raise ValueError(f"{backbone} is not a supported backbone.")

    return recipro_cam_feature_layer

