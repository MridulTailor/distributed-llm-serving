from pathlib import Path

import onnx
from onnx import TensorProto, helper

output = Path(__file__).parents[1] / "model_repository/echo/1/model.onnx"
output.parent.mkdir(parents=True, exist_ok=True)
x = helper.make_tensor_value_info("INPUT0", TensorProto.FLOAT, [None, 1])
y = helper.make_tensor_value_info("OUTPUT0", TensorProto.FLOAT, [None, 1])
scale = helper.make_tensor("scale", TensorProto.FLOAT, [1], [2.0])
graph = helper.make_graph(
    [helper.make_node("Mul", ["INPUT0", "scale"], ["OUTPUT0"])],
    "cpu_echo",
    [x],
    [y],
    [scale],
)
onnx.save(
    helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 13)],
    ),
    output,
)
print(f"wrote {output}")
