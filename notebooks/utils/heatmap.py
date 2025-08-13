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

    heatmap_feature_layer_lookup = {
        "resnet18": "/backbone/layer4/layer4.1/Add_quant",
        "resnet50": "/backbone/layer4/layer4.2/Add_quant",
    }
    heatmap_feature_layer = heatmap_feature_layer_lookup.get(backbone)
    if not heatmap_feature_layer:
        supported_backbones = ", ".join(heatmap_feature_layer_lookup.keys())
        raise ValueError(
            f"'{backbone}' is not a supported backbone. Supported backbones are: {supported_backbones}"
        )
    if not quantization:
        heatmap_feature_layer = heatmap_feature_layer.replace("_quant", "")

    return heatmap_feature_layer