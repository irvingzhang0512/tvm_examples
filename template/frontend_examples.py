from tvm import relay


def mxnet_to_relay(model_prefix,
                   epoch,
                   dtype="float32",
                   input_name="data",
                   input_shape=(1, 3, 112, 112)):
    import mxnet as mx

    shape_dict = {input_name: input_shape}
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    mod, params = relay.frontend.from_mxnet(sym, shape_dict, dtype, arg_params,
                                            aux_params)
    return mod, params


def pytorch_to_relay(pytorch_module,
                     dtype="float32",
                     input_name="data",
                     input_shape=(1, 3, 112, 112)):
    import torch
    pytorch_module = pytorch_module.eval()
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(pytorch_module, input_data).eval()
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model,
                                              shape_list,
                                              default_dtype=dtype)
    return mod, params
