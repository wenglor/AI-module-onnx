"""Evaluate the last layers of a Rotated RTMDet box branch only at the kept candidates."""

from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


@dataclass
class SparseBoxBranchResult:
    new_nodes: list = field(
        default_factory=list
    )  # names of the added nodes (keep them in float32)
    feature_convs: list = field(
        default_factory=list
    )  # convs whose outputs feed the gather, per level


def _consumers(model):
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    return consumers


def _producers(model):
    return {out: node for node in model.graph.node for out in node.output}


def _constant_value(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            return numpy_helper.to_array(node.attribute[0].t)
    raise ValueError(f"{name} is not a constant")


def _attr(node, name, default):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def _follow(consumers, tensor, op_types):
    """Follows single-consumer chains through the given op types; returns the first other consumer."""
    while True:
        users = consumers.get(tensor, [])
        if len(users) != 1:
            raise ValueError(f"expected one consumer of {tensor}, found {len(users)}")
        if users[0].op_type not in op_types:
            return users[0], tensor
        tensor = users[0].output[0]


def _prune_and_sort(model):
    from utils.quantization import sort_nodes_topologically

    sort_nodes_topologically(model)
    graph = model.graph
    needed = {o.name for o in graph.output}
    keep = []
    for node in reversed(list(graph.node)):
        if any(o in needed for o in node.output):
            keep.append(node)
            needed.update(i for i in node.input if i)
    keep.reverse()
    del graph.node[:]
    graph.node.extend(keep)
    used = {i for n in graph.node for i in n.input}
    for init in [i for i in graph.initializer if i.name not in used]:
        graph.initializer.remove(init)
    produced = {o for n in graph.node for o in n.output}
    for info in [
        v for v in graph.value_info if v.name not in used and v.name not in produced
    ]:
        graph.value_info.remove(info)


def compute_box_branch_at_kept_candidates(
    model: onnx.ModelProto, prefix: str = "sparse_box"
):
    """Rewrites a float32 Rotated RTMDet export in place (see the module docstring).

    Returns a SparseBoxBranchResult. Raises ValueError if the graph doesn't have the expected
    structure (then the model is left unchanged).
    """
    inferred = shape_inference.infer_shapes(model)
    shapes = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in list(inferred.graph.value_info) + list(inferred.graph.input)
    }
    inits = {i.name: i for i in model.graph.initializer}
    consumers, producers = _consumers(model), _producers(model)

    def weight(name):
        return numpy_helper.to_array(inits[name]).astype(np.float32)

    levels = []
    while True:
        lvl = len(levels)
        wname = f"bbox_head.reg_convs.{lvl}.1.conv.weight"
        convs = [
            n for n in model.graph.node if n.op_type == "Conv" and n.input[1] == wname
        ]
        if not convs:
            break
        (conv,) = convs
        w = weight(wname)
        if (
            w.shape[2:] != (3, 3)
            or _attr(conv, "group", 1) != 1
            or list(_attr(conv, "pads", [0] * 4)) != [1, 1, 1, 1]
            or list(_attr(conv, "strides", [1, 1])) != [1, 1]
            or list(_attr(conv, "dilations", [1, 1])) != [1, 1]
        ):
            raise ValueError(
                f"{conv.name}: expected a 3x3 conv with stride 1 and padding 1"
            )
        feature_relu = producers.get(conv.input[0])
        if feature_relu is None or feature_relu.op_type != "Relu":
            raise ValueError(f"{conv.name}: input is not a ReLU output")
        (relu,) = consumers[conv.output[0]]
        if relu.op_type != "Relu":
            raise ValueError(f"{conv.name}: not followed by a ReLU")
        heads = {
            n.input[1]: n for n in consumers[relu.output[0]] if n.op_type == "Conv"
        }
        reg = heads.get(f"bbox_head.rtm_reg.{lvl}.weight")
        ang = heads.get(f"bbox_head.rtm_ang.{lvl}.weight")
        if reg is None or ang is None or len(consumers[relu.output[0]]) != 2:
            raise ValueError(
                f"level {lvl}: expected exactly rtm_reg and rtm_ang after {relu.name}"
            )
        (exp,) = consumers[reg.output[0]]
        (mul,) = consumers[exp.output[0]]
        if exp.op_type != "Exp" or mul.op_type != "Mul":
            raise ValueError(f"level {lvl}: expected Exp -> Mul(stride) after rtm_reg")
        stride = float(
            _constant_value(model, next(i for i in mul.input if i != exp.output[0]))
        )
        _, h, w_ = shapes[conv.input[0]][1:]
        levels.append(
            dict(
                conv=conv,
                feature=conv.input[0],
                feature_conv=producers[feature_relu.input[0]],
                reg=reg,
                ang=ang,
                mul=mul,
                stride=stride,
                size=(h, w_),
                channels=w.shape[1],
            )
        )
    if not levels:
        raise ValueError("no bbox_head.reg_convs.*.1 convs found")

    # Where the post-processing reads the box distances and angles: per-level outputs are
    # flattened (Transpose, Reshape), concatenated over the levels, and gathered at the kept indices.
    def gather_of(level_outputs):
        concats = []
        for tensor in level_outputs:
            node, t = _follow(consumers, tensor, {"Transpose", "Reshape"})
            if node.op_type != "Concat":
                raise ValueError(f"expected a Concat over the levels after {tensor}")
            concats.append((node, list(node.input).index(t)))
        concat = concats[0][0]
        if any(c is not concat for c, _ in concats) or [k for _, k in concats] != list(
            range(len(levels))
        ):
            raise ValueError("per-level outputs are not concatenated in level order")
        (gather,) = consumers[concat.output[0]]
        if gather.op_type != "GatherND" or gather.input[0] != concat.output[0]:
            raise ValueError("expected GatherND at the kept indices after the Concat")
        return gather

    gather_reg = gather_of([lv["mul"].output[0] for lv in levels])
    gather_ang = gather_of([lv["ang"].output[0] for lv in levels])
    if gather_reg.input[1] != gather_ang.input[1]:
        raise ValueError("box and angle are gathered at different indices")
    kept = gather_reg.input[1]  # (K, 1) int64, ascending (from NonZero)

    new_nodes, new_inits = [], []

    def const(name, value):
        new_inits.append(numpy_helper.from_array(value, f"{prefix}_{name}"))
        return f"{prefix}_{name}"

    def node(op, inputs, name, **attrs):
        out = f"{prefix}_{name}"
        new_nodes.append(helper.make_node(op, inputs, [out], name=out, **attrs))
        return out

    # Tap table: for every candidate, the column of each of its 9 neighbours in the feature bank
    # (all levels' feature maps side by side, plus one zero column for the padding).
    sizes = [lv["size"] for lv in levels]
    offsets = np.cumsum([0] + [h * w for h, w in sizes])
    total = int(offsets[-1])
    taps = np.full((total, 9), total, np.int64)
    level_of = np.zeros(total, np.int64)
    for lvl, (h, w) in enumerate(sizes):
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        idx = offsets[lvl] + ys * w + xs
        level_of[idx.ravel()] = lvl
        for k, (dy, dx) in enumerate(
            (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
        ):
            y, x = ys + dy, xs + dx
            inside = (y >= 0) & (y < h) & (x >= 0) & (x < w)
            taps[idx[inside], k] = offsets[lvl] + y[inside] * w + x[inside]

    channels = levels[0]["channels"]
    columns = [
        node(
            "Reshape",
            [
                lv["feature"],
                const(f"flat_shape{lvl}", np.array([channels, -1], np.int64)),
            ],
            f"flat{lvl}",
        )
        for lvl, lv in enumerate(levels)
    ]
    columns.append(const("zero_column", np.zeros((channels, 1), np.float32)))
    bank = node("Concat", columns, "bank", axis=1)
    indices = node(
        "Squeeze", [kept, const("axis1", np.array([1], np.int64))], "indices"
    )
    kept_level = node(
        "Gather", [const("level_of", level_of), indices], "kept_level", axis=0
    )
    kept_taps = node("Gather", [const("taps", taps), indices], "kept_taps", axis=0)

    outputs = []
    for lvl, lv in enumerate(levels):
        w1 = weight(lv["conv"].input[1]).transpose(2, 3, 1, 0).reshape(9 * channels, -1)
        b1 = weight(lv["conv"].input[2])
        w2 = np.concatenate(
            [
                weight(lv["reg"].input[1])[:, :, 0, 0],
                weight(lv["ang"].input[1])[:, :, 0, 0],
            ]
        ).T
        b2 = np.concatenate([weight(lv["reg"].input[2]), weight(lv["ang"].input[2])])
        n_dist = weight(lv["reg"].input[1]).shape[0]

        mask = node(
            "Equal",
            [kept_level, const(f"level{lvl}", np.array(lvl, np.int64))],
            f"mask{lvl}",
        )
        t = node("Compress", [kept_taps, mask], f"taps{lvl}", axis=0)
        p = node("Gather", [bank, t], f"patches{lvl}", axis=1)  # (C, K_l, 9)
        p = node("Transpose", [p], f"patches_t{lvl}", perm=[1, 2, 0])  # (K_l, 9, C)
        p = node(
            "Reshape",
            [p, const(f"patch_shape{lvl}", np.array([-1, 9 * channels], np.int64))],
            f"patch_rows{lvl}",
        )
        f = node(
            "MatMul", [p, const(f"w1_{lvl}", np.ascontiguousarray(w1))], f"conv{lvl}"
        )
        f = node("Add", [f, const(f"b1_{lvl}", b1)], f"conv_bias{lvl}")
        f = node("Relu", [f], f"relu{lvl}")
        o = node(
            "MatMul",
            [f, const(f"w2_{lvl}", np.ascontiguousarray(w2.astype(np.float32)))],
            f"pred{lvl}",
        )
        o = node(
            "Add", [o, const(f"b2_{lvl}", b2.astype(np.float32))], f"pred_bias{lvl}"
        )
        dist = node(
            "Slice",
            [
                o,
                const(f"s0_{lvl}", np.array([0])),
                const(f"e0_{lvl}", np.array([n_dist])),
                const(f"a_{lvl}", np.array([1])),
            ],
            f"dist_raw{lvl}",
        )
        dist = node("Exp", [dist], f"dist_exp{lvl}")
        dist = node(
            "Mul",
            [dist, const(f"stride{lvl}", np.array(lv["stride"], np.float32))],
            f"dist{lvl}",
        )
        angle = node(
            "Slice",
            [
                o,
                const(f"s1_{lvl}", np.array([n_dist])),
                const(f"e1_{lvl}", np.array([n_dist + 1])),
                const(f"b_{lvl}", np.array([1])),
            ],
            f"angle{lvl}",
        )
        outputs.append((dist, angle))
    # levels are contiguous in the candidate order and the indices are ascending, so concatenating
    # the per-level results gives the candidates in the original order
    dist = node("Concat", [d for d, _ in outputs], "dist", axis=0)
    angle = node("Concat", [a for _, a in outputs], "angle", axis=0)

    for old, new in ((gather_reg.output[0], dist), (gather_ang.output[0], angle)):
        for n in model.graph.node:
            for k, name in enumerate(n.input):
                if name == old:
                    n.input[k] = new
    model.graph.node.remove(gather_reg)
    model.graph.node.remove(gather_ang)
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_inits)
    _prune_and_sort(model)
    return SparseBoxBranchResult(
        new_nodes=[n.name for n in new_nodes],
        feature_convs=[lv["feature_conv"].name for lv in levels],
    )


def quantize_features_with_relu_range(model: onnx.ModelProto, conv_names):
    """After quantize_static: quantize the given Conv (+ ReLU) outputs with the ReLU output range.

    The quantizer only merges a ReLU into the output quantization of the preceding Conv when the
    ReLU output feeds a quantized node. The features gathered by
    compute_box_branch_at_kept_candidates feed float32 nodes, so the Conv output is quantized before
    the ReLU, and half of the 8-bit range is spent on negative values the ReLU removes. This sets
    those quantization parameters to the ReLU range (zero point 0, same maximum), which is what the
    quantizer does for a merged Conv + ReLU: the quantization itself then clamps at zero. Modifies
    ``model`` in place and returns the number of changed tensors.
    """
    inits = {i.name: i for i in model.graph.initializer}
    consumers = _consumers(model)
    changed = 0
    for conv in [n for n in model.graph.node if n.name in set(conv_names)]:
        quant = [
            n
            for n in consumers.get(conv.output[0], [])
            if n.op_type == "QuantizeLinear"
        ]
        if not quant:
            continue  # ReLU already merged into the output quantization
        (quant,) = quant
        scale_name, zero_name = quant.input[1], quant.input[2]
        users = [
            n for n in model.graph.node if scale_name in n.input or zero_name in n.input
        ]
        if any(u.input[1] != scale_name or u.input[2] != zero_name for u in users):
            raise ValueError(
                f"{scale_name} / {zero_name} are shared with other tensors"
            )
        zero = numpy_helper.to_array(inits[zero_name])
        if zero.dtype != np.uint8:
            raise ValueError(f"{quant.name}: expected uint8 activations")
        scale = float(numpy_helper.to_array(inits[scale_name]))
        new_scale = (255 - int(zero)) * scale / 255
        inits[scale_name].CopyFrom(
            numpy_helper.from_array(np.array(new_scale, np.float32), scale_name)
        )
        inits[zero_name].CopyFrom(
            numpy_helper.from_array(np.array(0, np.uint8), zero_name)
        )
        changed += 1
    return changed
