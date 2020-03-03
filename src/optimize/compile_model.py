import tvm
from tvm import relay
from PIL import Image

from tvm.relay.frontend.pytorch import get_graph_input_names
from tvm.contrib import util

import torch
from src.core.model import model
from src.core.data import CustomTransforms

net, _, _, _ = model(torch.device('cpu'))
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(net, input_data).eval()

img_path = '../../data/Mathias/000111.jpg'
transforms = CustomTransforms(244,1.3)

img = Image.open(img_path)
img = transforms(img)
img = img.unsqueeze(0)

input_name = 'input.1'
shape_dict = {input_name: img.shape}
mod, params = relay.frontend.from_pytorch(scripted_model,
                                          shape_dict)

target = tvm.target.arm_cpu('rasp3b')

with relay.build_config(opt_level=3):
    mod, params = relay.optimize(mod, target, params)
    graph, lib, params = relay.build(mod,
                                     target=target,
                                     params=params)

lib_fname = ('../../models/tvm/net.tar')
lib.export_library(lib_fname)

with open('../../models/tvm/net.json', 'w') as f:
    f.write(graph)

with open('../../models/tvm/net.params', 'wb') as f:
    f.write(relay.save_param_dict(params))