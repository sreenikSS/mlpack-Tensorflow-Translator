# mlpack-TensorFlow Translator

- **Primary Dependencies : boost, armadillo, ensmallen, mlpack**
- C++11 is the minimum version required for cpp.

## Installing primary dependencies on MacOS

- Using Homebrew

  ```bash
  brew install boost armadillo ensmallen mlpack
  ```

---

## How to run the test_converter.cpp on MacOS

- First install libtorch.

  ```bash
  brew install libtorch
  ```

- Next, compile the file using the following command.

  ```bash
  g++ src/test_converter.cpp -o test_converter -lboost_serialization -lboost_program_options -larmadillo -lmlpack -lc10 -ltorch_cpu -std=c++14 -stdlib=libc++ -I /usr/local/Cellar/libtorch/1.6.0_1/include/torch/csrc/api/include
  ```

- Now, run the executable produced and log the output to a text file.

  ```bash
  ./test_converter >test_converter_output.txt
  ```

## How to run the test_converter.cpp on Linux

- Install the same dependencies as above and compile using the command below.

  ```bash
  g++ -o test_converter src/test_converter.cpp -I /usr/include/torch/csrc/api/include -lboost_serialization -lboost_program_options -larmadillo -lopenblas -fopenmp -lmlpack -ltorch -lc10 -ltorch_cpu
  ```

- Now, run the executable produced and log the output to a text file.

  ```bash
  ./test_converter >test_converter_output.txt
  ```

## Compiling onnx_to_mlpack.hpp on MacOS

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
