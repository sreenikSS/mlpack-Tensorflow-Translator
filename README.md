# mlpack-TensorFlow Translator

<div align="center">
<img src="docs/imgs/translator.jpeg" height="25%" width="25%">
<p></p>
</div>

## Dependencies

- **Primary Dependencies : boost, armadillo, ensmallen, mlpack**
- C++11 is the minimum version required for cpp.

## Functionality

This library can do two types of conversions currently.

- Converting mlpack models to torch (See example in mlpack_to_torch_test.cpp file)
- Converting onnx models to mlpack (See example in onnx_to_mlpack_test.cpp file)

This library also includes a header file only implementation of a standalone
model parser that can parse json files containing user-defined model details
to train neural networks.

## Installing primary dependencies on MacOS

- Using Homebrew

  ```bash
  brew install boost armadillo ensmallen mlpack
  ```

---

## How to run the mlpack_to_torch_test.cpp file on MacOS

- First install libtorch.

  ```bash
  brew install libtorch
  ```

- Next, compile the file using the following command.

  ```bash
  g++ tests/mlpack_to_torch_test.cpp -o tests/test_converter -lboost_serialization -lboost_program_options -larmadillo -lmlpack -lc10 -ltorch_cpu -std=c++14 -stdlib=libc++ -I /usr/local/Cellar/libtorch/1.6.0_1/include/torch/csrc/api/include -I src
  ```

- Now, run the executable produced and log the output to a text file.

  ```bash
  ./tests/test_converter >tests/test_converter_output.txt
  ```

## How to run the mlpack_to_torch_test.cpp file on Linux

- Install the same dependencies as above and compile using the command below.

  ```bash
  g++ -o tests/test_converter tests/mlpack_to_torch_test.cpp -I src -I /usr/include/torch/csrc/api/include -lboost_serialization -lboost_program_options -larmadillo -lopenblas -fopenmp -lmlpack -ltorch -lc10 -ltorch_cpu
  ```

- Now, run the executable produced and log the output to a text file.

  ```bash
  ./tests/test_converter >tests/test_converter_output.txt
  ```

## Compiling the onnx_to_mlpack.hpp file on MacOS

The only purpose of compiling the header file is to see whether the
functionality in it compiles correctly without giving any errors. Compiling this
using the command below will produce a GCC precompiled header file named onnx_to_mlpack.hpp.gch

- First install protobuf.

  ```bash
  brew install protobuf
  ```

- Then, install onnx in any directory
  (here, everything is done in the Downloads folder).

  ```bash
  cd Downloads
  git clone https://github.com/onnx/onnx.git
  cd onnx
  git submodule update --init --recursive
  python setup.py install
  ```

- Next, compile the file using the following command.

  ```bash
  g++ src/onnx_to_mlpack.hpp -I /Users/anjishnu/Downloads/onnx/.setuptools-cmake-build/ -DONNX_ML=1 -I /usr/local/bin/protoc -std=c++14 -stdlib=libc++ -lboost_serialization -lboost_program_options -larmadillo -lmlpack
  ```

## How to run the onnx_to_mlpack_test.cpp file on MacOS

- This file just #includes onnx_to_mlpack.hpp and has an empty main function.
- Compiling this file with this command gives a linker error presently.

  ```bash
  g++ tests/onnx_to_mlpack_test.cpp -I /Users/anjishnu/Downloads/onnx/.setuptools-cmake-build/ -DONNX_ML=1 -I /usr/local/bin/protoc -std=c++14 -stdlib=libc++ -lboost_serialization -lboost_program_options -larmadillo -lmlpack -I src
  ```

---

Some notes regarding future development tasks:

## mlpack layers that need some modification

- [x] BatchNorm - Missing argument `momentum` ✅
- [ ] MaxPool - Missing argument `pads` ❌
- [ ] Convolution - Missing argument `group` ❌
- [ ] Selu (elu) - Argument `lambda` (called gamma in onnx) is not modifiable ❌

## mlpack layers that need to be added

- [x] Softmax - We have LogSoftmax and not softmax but many of the popular
   pre-trained models in the onnx zoo use softmax ✅
- [ ] LRN - The AlexNet model of onnx uses LRN, so it's a must if we are to support
   AlexNet ❌

**Reference to the onnx models can be found at**
<https://github.com/onnx/onnx/blob/master/docs/Operators.md>

---
