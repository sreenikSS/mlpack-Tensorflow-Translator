# model conversion examples

This folder currently includes the following :

- ```Digit recognizer.ipynb``` which generates the ```onnx_conv_model.onnx``` file.

- ```Model_testing.ipynb``` which loads the ```onnx_conv_model.onnx``` file and
  explores different aspects of it.

- Two other onnx_model files, namely ```mnist_model.onnx``` and ```onnx_linear_model.onnx```.

- Contributions of other models are welcome to this folder. Read below.

## Testing new models

Onnx is not too consistent with its conversions not to mention its coherency
across frameworks. I have developed the translator keeping in mind only a
few models that I have created.

Hence, contributions are welcome in the form of (somewhat pre-trained) models,
preferably convolutional. Supported model frameworks are:

- Tensorflow
- Keras
- Onnx
- Pytorch
- Caffe
- Chainer

It would be convenient if the code used to create the model is also uploaded
(maybe in the form of the original source or just a screenshot of the model
building code) due to the exact information it can provide.

Each and every contribution can have a pivotal role to play in the further
development and enhanced compatibility of the model translator.
Thank you for reading and looking forward to your contribution :)

## Dependencies

The only dependencies for running the ipython notebooks in this folder are
the following python packages.

- ```Keras```
- ```onnx```
- ```onnxmltools```
- ```numpy```

## Installing Dependencies

It is preferable to install these packages in a virtual environment.
To create a virtual environment, [download and install](https://docs.conda.io/en/latest/miniconda.html)
the suitable version of miniconda for your system.

```bash
# Step-1: Create virtual anaconda environment.
conda create -n mlpack-translator python=3.8
conda activate mlpack-translator

# Step-2: Install dependencies.
pip install numpy
pip install Keras
pip install onnx
pip install onnxmltools
```

Once the dependencies are installed, the notebooks can be executed in the
environment containing those packages.
