#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <iostream>

int main() {
  DLDevice dev{kDLCPU, 0};
  std::string paht_to_lib = "../lib/cpu.so";
  std::vector<int64_t> input_shape = {1, 3, 112, 112};
  std::vector<int64_t> output_shape = {1, 128};
  char *input_blob_name = "data";

  tvm::runtime::Module mod_factory =
      tvm::runtime::Module::LoadFromFile(paht_to_lib);

  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty(
      input_shape, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty(
      output_shape, DLDataType{kDLFloat, 32, 1}, dev);

  // init input array
  int input_size =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  for (int i = 0; i < input_size; ++i) {
    static_cast<float *>(x->data)[i] = 1.;
  }

  set_input(input_blob_name, x);
  run();
  get_output(0, y);

  // print the first 10 elements of results
  for (int i = 0; i < 10; ++i)
    std::cout << static_cast<float *>(y->data)[i] << " ";
  std::cout << std::endl;
}