## Insightface-Arcface to TVM

## Steps

+ Step 1: Download arcface pretrained models from [model zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) and unzip them.
+ Step 2: Modify `model_prefix`, `epoch`, etc in `python/main.py`. Then run this script to
  + Convert downloaded model to tvm relay.
  + Auto Tune.
  + Inference with tvm.
  + Verify results with mxnet.
+ Step 3: Deploy with C++.
  + Modify `TVM_ROOT` in `CMakeLists.txt`.
  + `mkdir build && cd build && cmake .. && make` and run `./main`

## TODO

+ [x] Auto tune.
+ [x] Deploy with Python.
+ [x] Deploy with C++.
+ [ ] Dynamic Batch Size.
  + TVM doesn't support dynamic batch size for now.
  + Try TVM-TensorRT
  + Try different models for different batch size.
