# model conversion examples

This folder currently includes the following :

- ```Digit recognizer.ipynb``` which generates the ```onnx_conv_model.onnx``` file.

- ```Model_testing.ipynb``` which loads the ```onnx_conv_model.onnx``` file and
  explores different aspects of it.

- Two other onnx_model files, namely ```mnist_model.onnx``` and ```onnx_linear_model.onnx```.

- Contributions of other models are welcome to this folder. Newer model will be
  added to the sub-folder ```new_models```, which has a README explaining the requirements.

## Dependencies

The only dependencies for running the ipython notebooks in this repository are
the following python packages.

- ```Keras```
- ```onnx```
- ```onnxmltools```
- ```numpy```

## Installing Dependencies

It is preferable to install these packages in a virtual environment.

```bash
pip install numpy
pip install Keras
pip install onnx
pip install onnxmltools
```
