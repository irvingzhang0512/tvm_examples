## Insightface-Arcface to TVM

## Steps

+ Step 1: Download arcface pretrained models from [model zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) and unzip them.
+ Step 2: Use `mxnet_to_tvm.py` to convert downloaded model to tvm relay.
+ Step 3: Auto tune and deploy.

## TODO

+ [ ] Auto tune.
+ [ ] Deploy with Python.
+ [ ] Deploy with C++.
+ [ ] Dynamic Batch Size.
  + TVM doesn't support dynamic batch size for now.
  + Try TVM-TensorRT
  + Try different models for different batch size.
