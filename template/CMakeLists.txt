cmake_minimum_required(VERSION 3.5)

project(tvm_arcface)

set(CMAKE_CXX_STANDARD 14)
set(TVM_ROOT "/ssd01/zhangyiyang/tvm")
set(DMLC_CORE "${TVM_ROOT}/3rdparty/dmlc-core")

include_directories(${TVM_ROOT}/include ${TVM_ROOT}/3rdparty/dmlc-core/include ${TVM_ROOT}/3rdparty/dlpack/include)
link_directories(${TVM_ROOT}/build)

add_executable(main src/main.cc)
target_link_libraries(main tvm_runtime)
