# mlpack layers that need some modification

1. Batchnorm - Missing argument `momentum`
2. Maxpool - Missing argument `pads`
3. Convolution - Missing argument `group`
4. Selu (elu) - Argument `lambda` (called gamma in onnx) is not modifiable

## mlpack layers that need to be added

1. Softmax - We have logsoftmax and not softmax but many of the popular pretrained models in the onnx zoo use softmax
2. LRN - The Alexnet model of onnx uses LRN, so it's a must if we are to support Alexnet

**Reference to the onnx models can be found at** <https://github.com/onnx/onnx/blob/master/docs/Operators.md>

## How to run the test_converter.cpp on MacOS

g++ -o test_converter src/test_converter.cpp -lboost_serialization
-lboost_program_options -larmadillo -lmlpack -lc10 -ltorch_cpu -std=c++14
-stdlib=libc++ -I
/usr/local/Cellar/libtorch/1.6.0_1/include/torch/csrc/api/include

## How to run the test_converter.cpp on Linux

g++ -o test_converter src/test_converter.cpp -I
/usr/include/torch/csrc/api/include -lboost_serialization
-lboost_program_options -larmadillo -lopenblas -fopenmp -lmlpack -ltorch -lc10
-ltorch -ltorch_cpu
