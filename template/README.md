# Template for Common TVM projects

## Basic Functions

+ Convert all kinds of models to relay(`python/frontend_exmamples.py`).
  + [x] MXNet
  + [x] PyTorch
  + [ ] ONNX
+ AutoTune(`python/tvm_development_utils.py`)
  + [x] Local AutoTune
  + [x] Remote AutoTune
+ Deployment
  + [x] Python Runtime(`python/tvm_deployment_utils.py`)
  + [x] C++ Runtime(`src/main.cc` and `CMakeLists.txt`)
+ [ ] Dynamic Batch Size.
  + TVM doesn't support dynamic batch size for now.
  + Try TVM-TensorRT.

## Code review

### `python/tvm_development_utils.py`

+ Functions: AutoTune(local/remote), inference, evaluate, etc.
+ Steps to use
  + Step 1: Overwrite abstract method `network_fn` and get an object.
    + Samples could be found in `python/frontend_examples.py`.
  + Step 2: AutoTune with Python API, get schedule.
    + `tool.local_auto_scheduler()`
  + Step 3: Inference/evaluate with Python API.
    + `tool.export_lib(target_lib_path)`
    + `tool.evaluate()`
    + `tool.inference(numpy_inputs, input_blob_name)`

### `python/tvm_deployment_utils.py`

+ Functions: Inference by exported library with Python API.
+ Steps to use
  + Step 1: Generate library with `python/tvm_development_utils.py`
  + Step 2: Modify params in `python/tvm_deployment_utils.py` and run.

### Inference with C++ API

+ Related codes
  + `src`
  + `CMakeLists.txt`
+ Steps to use
  + Step 1: Modify params in `src/main.cc`.
  + Step 2: Build project with `mkdir build && cd build && cmake .. && make`
  + Step 3: Run `./main`
+ Notes:
  + Python and C++ API are almost the same.
