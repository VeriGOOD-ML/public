import argparse
from onnx import ModelProto, GraphProto
import onnx
from pathlib import Path
import polymath as pm


MODEL_DIR = Path(f"{Path(__file__).parent}/models")
LAYER_DIR = Path(f"{Path(__file__).parent}/layers")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_onnx_model(model_name):
    onnx_model = onnx.load(model_name)
    return onnx_model

def store_layer(layer_name, model_name, model: ModelProto = None):
    layer_path = f"{LAYER_DIR}/{layer_name}.onnx"
    model_path = f"{MODEL_DIR}/{model_name}.onnx"

    inputs = []
    outputs = []
    if model is None:
        model = onnx.load_model(model_path)

    for n in model.graph.node:
        if n.name == layer_name:
            inputs = n.input
            outputs = n.output
            break

    onnx.utils.extract_model(model_path, layer_path, inputs, outputs)


def convert_model_to_polymath(model_path):
    graph = pm.from_onnx(model_path)
    root_path = Path(model_path).parent
    pm.pb_store(graph, f"{root_path}/srdfg/")


def store_unique_model_layers(model_name, store_as_polymath=False):
    layers = {}
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    model = onnx.load_model(model_path)
    for n in model.graph.node:
        if n.op_type not in layers:
            inputs = n.input
            outputs = n.output
            op_name = n.op_type.lower()
            layer_path = f"{LAYER_DIR}/{model_name}_{op_name}.onnx"

            onnx.utils.extract_model(model_path, layer_path, inputs, outputs)
            if store_as_polymath:
                convert_model_to_polymath(layer_path)
            layers[n.op_type] = 1
        else:
            layers[n.op_type] += 1

def store_target_model_layer(model_name, layer_name, store_name=None, store_as_polymath=False):
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    model = onnx.load_model(model_path)
    found = False
    for n in model.graph.node:
        if n.op_type == layer_name:
            inputs = n.input
            outputs = n.output
            op_name = n.op_type.lower() if store_name is None else store_name
            layer_path = f"{LAYER_DIR}/{model_name}_{op_name}.onnx"

            onnx.utils.extract_model(model_path, layer_path, inputs, outputs)
            if store_as_polymath:
                convert_model_to_polymath(layer_path)
            found = True
            break

    if not found:
        raise RuntimeError(f"Unable to find layer {layer_name} in model")


if __name__ == "__main__":
    # argparser = argparse.ArgumentParser(description='ONNX Benchmark Generator')
    # argparser.add_argument('-b', '--benchmark', required=True,
    #                        help='Name of the benchmark to create. One of "resnet18", "lenet')
    #
    #
    # argparser.add_argument('-t', '--training_mode', type=str2bool, nargs='?', default=False,
    #                        const=True, help='Whether or not the model is in training mode')
    #
    #
    # argparser.add_argument('-pm', '--to_polymath', type=str2bool, nargs='?', default=False,
    #                        const=True, help='Whether or not the model should be converted to PolyMath')
    # args = argparser.parse_args()
    model_name = 'resnet18_train'
    model_path = f"{MODEL_DIR}/{model_name}.onnx"

    # convert_model_to_polymath(model_path)
    # store_unique_model_layers(model_name, store_as_polymath=True)
    store_target_model_layer(model_name, "Conv", store_name="conv_bias", store_as_polymath=True)
