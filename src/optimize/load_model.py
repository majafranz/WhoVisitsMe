from tvm.contrib import graph_runtime
import tvm
from tvm import relay
from os import path

input_name = 'input.1'

def load_module():
    tvm_model_dir = path.join('..', 'models', 'tvm')
    lib = tvm.runtime.load_module(path.join(tvm_model_dir,'net.tar'))
    graph = open(path.join(tvm_model_dir, 'net.json')).read()
    params = relay.load_param_dict(bytearray(open(path.join(tvm_model_dir, 'net.params'), 'rb').read()))

    ctx = tvm.cpu(0)
    module = graph_runtime.create(graph, lib, ctx)

    module.set_input(**params)
    return module
