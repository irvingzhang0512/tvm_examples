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
