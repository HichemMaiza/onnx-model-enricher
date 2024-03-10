import onnx
from onnx.checker import check_model
from onnx.helper import make_node, make_model, make_graph, make_tensor, make_tensor_value_info

from constants import MODEL_PATH, UNET_INPUT_SHAPE, OUTPUT_MODEL_PATH

model = onnx.load(MODEL_PATH)

# the image tensor is of shape [batch_size, float_array]
input_tensor = make_tensor_value_info('image_tensor', onnx.TensorProto.FLOAT, [-1, -1])

# this is a constant tensor to be passed to the reshape operation
shape_tensor_info = make_tensor_value_info('shape_tensor', onnx.TensorProto.INT64, [len(UNET_INPUT_SHAPE)])
shape_tensor = make_tensor('shape_tensor', onnx.TensorProto.INT64, [4], [1] + UNET_INPUT_SHAPE)

# the reshape operation input takes two inputs variable, check onnx documentation
node_1 = make_node(
    "Reshape",
    inputs=["image_tensor", "shape_tensor"],
    outputs=["input.1"]
)

# create the new stack of nodes
nodes = [node_1] + [node for node in model.graph.node]

# a very important step is to initialize the weights and bias of the new model !
# these weights must be recopied from the old model to the new one
# we also need to initialize the shape_tensor, because we want to be a constant
initializers = model.graph.initializer[:] + [shape_tensor]

# create the new graph
new_graph = make_graph(nodes, "enriched_graph", [input_tensor], model.graph.output, initializer=initializers)

# construct the new model
new_model = make_model(new_graph, opset_imports=[onnx.helper.make_opsetid("", 12)])

# validate the model
check_model(new_model)

# save the model
onnx.save(new_model, OUTPUT_MODEL_PATH)