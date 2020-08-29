# mlpack-TensorFlow Translator

<div align="center">
<img src="docs/imgs/translator.jpeg" height="25%" width="25%">
<p></p>
</div>

## Dependencies

- **Primary Dependencies : boost, armadillo, ensmallen, mlpack**
- C++11 is the minimum version required for cpp.

## Functionality

The name of the repository is a misnomer as none of the functionalities have
anything to do with Tensorflow directly. However, it allows
to convert onnx models to mlpack and any Tensorflow model can be
converted into onnx using [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow).

This library can do two types of conversions currently.

- Converting mlpack models to torch (See example in mlpack_to_torch_test.cpp file)
- Converting onnx models to mlpack (See example in onnx_to_mlpack_test.cpp file)

This library also includes a header file only implementation of a standalone
model parser that can parse json files containing user-defined model details
to train neural networks.

For the conversion process, the library currently has tested support for
linear and convolutional layers. Further support can be added as and when
more examples are available in the ```examples``` repository.

## Running the test cases

### Installing primary dependencies on MacOS

- Using Homebrew

  ```bash
  brew install boost armadillo ensmallen mlpack
  ```

---

### How to run the mlpack_to_torch_test.cpp file on MacOS

- First install libtorch.

  ```bash
  brew install libtorch
  ```

- Next, compile the file using the following command.

  ```bash
  g++ -w tests/mlpack_to_torch_test.cpp -o tests/mlpack_to_torch_test -lboost_serialization -lboost_program_options -larmadillo -lmlpack -lc10 -ltorch_cpu -std=c++14 -stdlib=libc++ -I /usr/local/Cellar/libtorch/1.6.0_1/include/torch/csrc/api/include -I src
  ```

- Now, run the executable produced and log the output to a text file.

  ```bash
  ./tests/mlpack_to_torch_test >tests/mlpack_to_torch_test_output.txt
  ```

### How to run the mlpack_to_torch_test.cpp file on Linux

- Install the same dependencies as above and compile using the command below.

  ```bash
  g++ -o tests/mlpack_to_torch_test -w tests/mlpack_to_torch_test.cpp -I src -I /usr/include/torch/csrc/api/include -lboost_serialization -lboost_program_options -larmadillo -lopenblas -fopenmp -lmlpack -ltorch -lc10 -ltorch_cpu
  ```

- Now, run the executable produced and log the output to a text file.

  ```bash
  ./tests/mlpack_to_torch_test >tests/mlpack_to_torch_test_output.txt
  ```

### How to run the onnx_to_mlpack_test.cpp file on MacOS

- This file just #includes onnx_to_mlpack.hpp and has an empty main function to
  test if there are any compilation errors on including the onnx_to_mlpack.hpp
  file.

- First install protobuf.

  ```bash
  brew install protobuf
  ```

- Then, install onnx in any easily accessible directory
  (here, everything is done in the Downloads folder).

  ```bash
  cd Downloads
  git clone https://github.com/onnx/onnx.git
  cd onnx
  git submodule update --init --recursive
  python setup.py install
  ```

- Next, compile this file with the following command.

  ```bash
  g++ -w tests/onnx_to_mlpack_test.cpp -L /Users/<username>/Downloads/onnx/.setuptools-cmake-build/ -DONNX_ML=1 -DONNX_NAMESPACE=onnx -L /usr/local/bin/ -std=c++14 -stdlib=libc++ -lboost_serialization -lboost_program_options -larmadillo -lmlpack -I src -I /Users/<username>/Downloads/onnx/.setuptools-cmake-build/ -lonnx_proto -lprotobuf -lpthread -o tests/onnx_to_mlpack_test
  ```

- Run the executable produced using the command below.

  ```bash
  ./tests/onnx_to_mlpack_test
  ```

---

## Some notes regarding future development tasks

### mlpack layers that need some modification

- [x] BatchNorm - Missing argument `momentum` ✅
- [ ] MaxPool - Missing argument `pads` ❌
- [ ] Convolution - Missing argument `group` ❌
- [ ] Selu (elu) - Argument `lambda` (called gamma in onnx) is not modifiable ❌

### mlpack layers that need to be added

- [x] Softmax - We have LogSoftmax and not softmax but many of the popular
   pre-trained models in the onnx zoo use softmax ✅
- [ ] LRN - The AlexNet model of onnx uses LRN, so it's a must if we are to support
   AlexNet ❌

**Reference to the onnx models can be found at**
<https://github.com/onnx/onnx/blob/master/docs/Operators.md>

---
