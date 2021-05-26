from abc import abstractmethod
import numpy as np
import os

import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class TvmDevelopmentUtils:
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

        try:
            self.mod, self.params = self.network_fn()
        except:
            logger.warning("self.mod and self.params are not initialized.")

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