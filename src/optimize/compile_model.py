import tvm
from tvm import relay
import torch
from os import path
from PIL import Image
from src.core.data import CustomTransforms
from src.core.model import model
from src.utils.config import LOAD_MODEL_PATH, SAVE_PATH, DATA_ROOT, IMAGE_SCALE, IMAGE_SIZE
from src.utils.logger import logger

def compile_model(load_path=LOAD_MODEL_PATH):
    net, _, _, _ = model(torch.device('cpu'), load_path=load_path)

    img_path = path.join(DATA_ROOT, 'None', '000000.jpg')
    transforms = CustomTransforms(IMAGE_SIZE, IMAGE_SCALE)

    img = Image.open(img_path)
    img = transforms(img)
    img = img.unsqueeze(0)

    scripted_model = torch.jit.trace(net, img).eval()
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

    tvm_model_dir = path.join(SAVE_PATH, 'tvm')

    save_name = load_path[:load_path.rfind('.')] + '{:s}' if load_path is not None else 'net{:s}'

    lib.export_library(path.join(tvm_model_dir, save_name.format('.tar')))

    with open(path.join(tvm_model_dir, save_name.format('.json')), 'w') as f:
        f.write(graph)

    with open(path.join(tvm_model_dir, save_name.format('.params')), 'wb') as f:
        f.write(relay.save_param_dict(params))

    logger.info('compiled model and saved it in {:s}'.format(path.join(tvm_model_dir, save_name.format(''))))