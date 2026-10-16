import torch


def compute_losses_in_float32(model):
    """Runs the loss computation of a detector's head in float32 under mixed precision.

    With mmengine's ``AmpOptimWrapper`` the whole training forward pass runs under
    ``torch.autocast``. The convolutions profit from float16, but the head's loss computation
    (label assignment with rotated IoU, box and angle decoding, the losses) is kept in float32:
    its inputs are cast to float32 and autocast is switched off for it.

    Call it once after the model is built (e.g. ``Runner.from_cfg``), before ``runner.train()``.

    Args:
        model: An mmdet / mmrotate detector with a ``bbox_head``.

    Returns:
        The model, with ``bbox_head.loss_by_feat`` wrapped.
    """
    head = model.bbox_head
    loss_by_feat = head.loss_by_feat

    def to_float32(x):
        if torch.is_tensor(x) and x.is_floating_point():
            return x.float()
        if isinstance(x, (list, tuple)):
            return type(x)(to_float32(v) for v in x)
        return x

    def loss_by_feat_float32(*args, **kwargs):
        with torch.autocast(device_type="cuda", enabled=False):
            return loss_by_feat(*to_float32(args), **kwargs)

    head.loss_by_feat = loss_by_feat_float32
    return model
