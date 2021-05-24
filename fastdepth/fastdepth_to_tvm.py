import numpy as np
import torch
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor

from fastdepth import get_scripted_moidel

INPUT_NAME = "input0"


def pytorch_to_tvm(scripted_model, input_shape):
    shape_list = [(INPUT_NAME, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod, params


def _test_fastdepthv2_tvm(scripted_model,
                          input_shape,
                          target=tvm.target.Target("llvm", host="llvm"),
                          dev=tvm.cpu(0),
                          dtype="float32"):
    mod, params = pytorch_to_tvm(scripted_model, input_shape)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    inputs = np.ones(input_shape, dtype=dtype)
    m = graph_executor.GraphModule(lib["default"](dev))
    m.set_input(INPUT_NAME, tvm.nd.array(inputs))
    m.run()
    tvm_output = m.get_output(0)
    return tvm_output.asnumpy()


if __name__ == '__main__':
    input_shape = (1, 3, 224, 224)

    scripted_model = get_scripted_moidel(
        'v2', '../data/fastdepth/FastDepthV2_L1GN_Best.pth')
    inputs = torch.FloatTensor(torch.ones(*input_shape))
    pytorch_output = scripted_model(inputs).cpu().detach().numpy()
    print("pytorch output: ", pytorch_output.reshape(-1)[:10])

    tvm_output = _test_fastdepthv2_tvm(scripted_model, input_shape)
    print("tvm output: ", tvm_output.reshape(-1)[:10])
