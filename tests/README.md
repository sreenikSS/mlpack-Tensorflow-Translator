# Test Cases

## Running the model generator file

```mlpack_model_generator.cpp``` is a simple file which
demonstrates how to create a XML file with initialized weights for a mlpack
model. Code for any other model in mlpack can be
similarly written and saved to a XML file. Then the translator can be used to
convert that into a ```.pt``` file for pytorch using the ```convertModel()```
function from ```mlpack_to_torch.hpp```

To compile and run this file, use the following commands.

```bash
# compile
g++ -w tests/mlpack_model_generator.cpp -o tests/mlpack_model_generator -lboost_serialization -lboost_program_options -larmadillo -lmlpack -lc10 -ltorch_cpu -std=c++14 -stdlib=libc++ -I /usr/local/Cellar/libtorch/1.6.0_1/include/torch/csrc/api/include -I src

# run
./tests/mlpack_model_generator
```

---

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
