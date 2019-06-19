## mlpack layers that need some modification
1. Batchnorm - Missing argument `momentum`
2. Maxpool - Missing argument `pads`
3. Convolution - Missing argument `group`
4. Selu (elu) - Argument `lambda` (called gamma in onnx) is not modifiable

## mlpack layers that need to be added
1. Softmax - We have logsoftmax and not softmax but many of the popular pretrained models in the onnx zoo use softmax
2. LRN - The Alexnet model of onnx uses LRN, so it's a must if we are to support Alexnet

**Reference to the onnx models can be found at** https://github.com/onnx/onnx/blob/master/docs/Operators.md
