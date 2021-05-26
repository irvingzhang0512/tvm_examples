import logging

import numpy as np
import tvm
from tvm.contrib import graph_executor
import mxnet as mx

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

INPUT_NAME = "data"


class TvmDeployementTool:
    def __init__(self, lib_path, dev=tvm.device("cuda", 0)):
        self.lib = tvm.runtime.load_module(lib_path)
        self.dev = dev

    @property
    def module(self):
        if getattr(self, '_module', None) is None:
            self._module = graph_executor.GraphModule(self.lib['default'](
                self.dev))
        return self._module

    def do_inference(self, inputs, input_name):
        data_tvm = tvm.nd.array(inputs)
        self.module.set_input(input_name, data_tvm)
        self.module.run()
        return self.module.get_output(0)

    def evaluate(self, repeat=3, min_repeat_ms=500):
        logger.info("Evaluate inference time cost...")
        ftimer = self.module.module.time_evaluator("run",
                                                   self.dev,
                                                   repeat=repeat,
                                                   min_repeat_ms=min_repeat_ms)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        logger.info("Mean inference time (std dev): %.2f ms (%.2f ms)" %
                    (np.mean(prof_res), np.std(prof_res)))


if __name__ == '__main__':
    tool = TvmDeployementTool(
        "/ssd01/zhangyiyang/tvm_examples/insightface/lib/cpu.so",
        tvm.device("cpu"))
    print(
        tool.do_inference(np.ones((1, 3, 112, 112), np.float32),
                          INPUT_NAME).asnumpy().reshape(-1)[:10])
