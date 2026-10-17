import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_dummy_model(path: Path, h=256, w=320):
    """Stand-in 'model' (one 1x1 conv): fires on dark pixels. Lets the whole pipeline be
    tested without PyTorch or trained weights."""
    W = numpy_helper.from_array(np.full((1, 3, 1, 1), -10 / 3, np.float32), "W")
    B = numpy_helper.from_array(np.array([-14.0], np.float32), "B")
    node = helper.make_node("Conv", ["input", "W", "B"], ["logits"])
    graph = helper.make_graph(
        [node], "dummy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, h, w])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, h, w])], [W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


@pytest.fixture()
def dummy_model(tmp_path):
    p = tmp_path / "dummy.onnx"
    build_dummy_model(p)
    return str(p)
