from tvm.contrib import graph_runtime
import tvm

input_name = 'input.1'

def load_module():
    lib = tvm.runtime.load_module('../../models/net.tar')
    graph = open('net.json').read()
    params = bytearray(open('net.params', 'rb').read())

    ctx = tvm.cpu(0)
    module = graph_runtime.create(graph, lib, ctx)

    module.set_input(**params)
    return module
