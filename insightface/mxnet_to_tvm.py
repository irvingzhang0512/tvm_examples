import mxnet as mx
import tvm.relay as relay

INPUT_NAME = "data"


def arcface_mxnet_to_tvm(model_prefix,
                         epoch,
                         image_size=(112, 112),
                         dtype="float32"):
    shape_dict = {INPUT_NAME: (1, 3) + image_size}
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    mod, params = relay.frontend.from_mxnet(sym, shape_dict, dtype, arg_params,
                                            aux_params)
    return mod, params


def _test_with_tvm(inputs,
                   model_prefix,
                   epoch,
                   image_size=(112, 112),
                   dtype="float32",
                   target="cuda"):
    import tvm
    from tvm.contrib import graph_runtime

    mod, params = arcface_mxnet_to_tvm(model_prefix, epoch, image_size, dtype)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod["main"], target, params=params)
    ctx = tvm.gpu(0)
    m = graph_runtime.GraphModule(lib["default"](ctx))
    # set inputs
    m.set_input(INPUT_NAME, tvm.nd.array(inputs.astype(dtype)))
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0)

    return tvm_output.asnumpy()


def _test_with_mxnet(inputs,
                     model_prefix,
                     epoch,
                     image_size=(112, 112),
                     ctx=mx.gpu(0)):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[(INPUT_NAME, (1, 3) + image_size)])
    model.set_params(arg_params, aux_params)
    data = mx.nd.array(inputs)
    db = mx.io.DataBatch(data=(data, ))
    model.forward(db, is_train=False)
    return model.get_outputs()[0].asnumpy()


if __name__ == '__main__':
    import numpy as np

    model_prefix = "/ssd01/zhangyiyang/tensorrtx/arcface/insightface/deploy/model-y1-test2/model"
    epoch = 0
    image_size = (112, 112)
    dtype = "float32"
    inputs = np.ones((1, 3) + image_size, dtype)

    print(
        _test_with_tvm(inputs, model_prefix, epoch, image_size,
                       dtype).reshape(-1)[:10])
    print(
        _test_with_mxnet(inputs, model_prefix, epoch,
                         image_size).reshape(-1)[:10])
