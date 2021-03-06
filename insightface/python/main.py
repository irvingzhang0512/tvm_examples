import logging
import os
from abc import abstractmethod

import mxnet as mx
import numpy as np
import tvm
from tvm import auto_scheduler, relay
from tvm.contrib import graph_executor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

INPUT_NAME = "data"


class BaseTvmUtils:
    def __init__(self,
                 network_name,
                 image_size,
                 target,
                 layout="NHWC",
                 dtype="float32",
                 log_file=None,
                 lib_path=None):
        # general args
        self.network_name = network_name
        self.batch_size = network_name[0]
        self.image_size = image_size
        self.target = target
        self.layout = layout
        self.dtype = dtype
        self.log_file = log_file if log_file is not None \
            else f"{network_name}-{image_size}-{layout}-{target.kind.name}.json"

        self.mod, self.params = self.network_fn()

        if lib_path is not None:
            self.deserialize_lib(lib_path)

    @property
    def dev(self):
        if getattr(self, '_dev', None) is None:
            self._dev = tvm.device(str(self.target), 0)
        return self._dev

    @property
    def lib(self):
        if getattr(self, '_lib', None) is None:
            if self.log_file is not None and os.path.exists(self.log_file):
                with auto_scheduler.ApplyHistoryBest(self.log_file):
                    with tvm.transform.PassContext(
                            opt_level=3,
                            config={"relay.backend.use_auto_scheduler": True}):
                        self._lib = relay.build(self.mod,
                                                target=self.target,
                                                params=self.params)
                    logger.info(f"load optimized library from {self.log_file}")
            else:
                with tvm.transform.PassContext(
                        opt_level=3,
                        config={"relay.backend.use_auto_scheduler": True}):
                    self._lib = relay.build(self.mod,
                                            target=self.target,
                                            params=self.params)
                    logger.info("load unoptimzed library")

        return self._lib

    @property
    def module(self):
        if getattr(self, '_module', None) is None:
            self._module = graph_executor.GraphModule(self.lib['default'](
                self.dev))
        return self._module

    @abstractmethod
    def network_fn(self):
        # returns (mod, params)
        pass

    def inference(self, inputs, input_name):
        data_tvm = tvm.nd.array(inputs)
        self.module.set_input(input_name, data_tvm)
        self.module.run()
        return self.module.get_output(0)

    def local_auto_scheduler(self,
                             repeat=1,
                             min_repeat_ms=300,
                             timeout=10,
                             num_measure_trials=200):
        # extract tasks
        tasks, task_weights = auto_scheduler.extract_tasks(
            self.mod["main"], self.params, self.target)
        for idx, task in enumerate(tasks):
            logger.debug("========== Task %d  (workload key: %s) ==========" %
                         (idx, task.workload_key))
            logger.debug(task.compute_dag)

        # generate tuner
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)

        logging.info("Begin tuning...")
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=repeat, min_repeat_ms=min_repeat_ms, timeout=timeout)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=num_measure_trials,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(self.log_file)],
        )
        tuner.tune(tune_option)

        # update self.lib
        with auto_scheduler.ApplyHistoryBest(self.log_file):
            with tvm.transform.PassContext(
                    opt_level=3,
                    config={"relay.backend.use_auto_scheduler": True}):
                self._lib = relay.build(self.mod,
                                        target=self.target,
                                        params=self.params)
            logger.info(f"load optimized library from {self.log_file}")

    def remote_auto_scheduler(self, device_key, rpc_host, rpc_port):
        # generate tasks
        tasks, task_weights = auto_scheduler.extract_tasks(
            self.mod["main"], self.params, self.target)
        for idx, task in enumerate(tasks):
            logger.debug("========== Task %d  (workload key: %s) ==========" %
                         (idx, task.workload_key))
            logger.debug(task.compute_dag)

        # generate tuner
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)

        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=200,
            builder=auto_scheduler.LocalBuilder(),
            runner=auto_scheduler.RPCRunner(
                device_key,
                host=rpc_host,
                port=rpc_port,
                timeout=30,
                repeat=1,
                min_repeat_ms=200,
                enable_cpu_cache_flush=True,
            ),
            measure_callbacks=[auto_scheduler.RecordToFile(self.log_file)],
        )
        tuner.tune(tune_option)

        # update self.lib
        with auto_scheduler.ApplyHistoryBest(self.log_file):
            with tvm.transform.PassContext(
                    opt_level=3,
                    config={"relay.backend.use_auto_scheduler": True}):
                self._lib = relay.build(self.mod,
                                        target=self.target,
                                        params=self.params)
            logger.info(f"load optimized library from {self.log_file}")

    def export_lib(self, lib_path):
        self.lib.export_library(lib_path)

    def deserialize_lib(self, lib_path):
        self._lib = tvm.runtime.load_module(lib_path)

    def evaluate(self, repeat=3, min_repeat_ms=500):
        logger.info("Evaluate inference time cost...")
        ftimer = self.module.module.time_evaluator("run",
                                                   self.dev,
                                                   repeat=repeat,
                                                   min_repeat_ms=min_repeat_ms)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        logger.info("Mean inference time (std dev): %.2f ms (%.2f ms)" %
                    (np.mean(prof_res), np.std(prof_res)))


class ArcFaceUtils(BaseTvmUtils):
    def __init__(self,
                 model_prefix,
                 epoch,
                 network_name,
                 image_size,
                 target,
                 layout="NHWC",
                 dtype="float32",
                 log_file=None):
        self.model_prefix = model_prefix
        self.epoch = epoch
        super().__init__(network_name, image_size, target, layout, dtype,
                         log_file)

    def network_fn(self):
        # returns (mod, params)
        shape_dict = {"data": self.image_size}
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            self.model_prefix, self.epoch)
        mod, params = relay.frontend.from_mxnet(sym, shape_dict, self.dtype,
                                                arg_params, aux_params)
        return mod, params


def _test_with_mxnet(inputs,
                     model_prefix,
                     epoch,
                     image_size=(1, 3, 112, 112),
                     ctx=mx.gpu(0)):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[(INPUT_NAME, image_size)])
    model.set_params(arg_params, aux_params)
    data = mx.nd.array(inputs)
    db = mx.io.DataBatch(data=(data, ))
    model.forward(db, is_train=False)
    return model.get_outputs()[0].asnumpy()


if __name__ == '__main__':
    model_prefix = "../../data/insightface/model-y1-test2/model"
    epoch = 0
    image_size = (1, 3, 112, 112)
    dtype = "float32"
    inputs = np.ones((1, 3) + image_size, dtype)

    # init
    tool = ArcFaceUtils(
        model_prefix=model_prefix,
        epoch=epoch,
        network_name='arcface-mobilefacenet',
        image_size=image_size,
        target=tvm.target.Target("llvm"),
        layout="NHWC",
        dtype=dtype,
        log_file='lib/arcface-mobilefacenet-(1, 3, 112, 112)-NHWC-llvm.json')

    # auto tune, generate log_file
    tool.local_auto_scheduler()

    # evaluate inference speed
    tool.evaluate()

    # export optimized library
    tool.export_lib("lib/cpu.so")

    # inference with tvm
    tool.inference(inputs, INPUT_NAME)

    # verify with mxnet
    _test_with_mxnet(inputs, model_prefix, epoch, image_size)
