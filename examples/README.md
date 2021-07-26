# model conversion examples

Code used for generating models is present in the ```code``` folder.
Models in ONNX format are present in the ```models``` folder.

This folder currently includes the following :

- ```Digit recognizer.ipynb``` which generates the ```onnx_conv_model.onnx```
  file.

- Two other onnx_model files, namely ```mnist_model.onnx``` and
  ```onnx_linear_model.onnx```.

- Contributions of other models are welcome to this folder. Read below for
  further details.

## Testing new models

Onnx is not too consistent with its conversions not to mention its coherency
across frameworks. The translator has been developed keeping in mind only a
few models that the core contributors have worked with for this repository,
and thus currently tests support for translation of
only linear and convolutional layers. As and when more models are added here,
we can test for a wider variety of layers.

Hence, contributions are welcome in the form of (somewhat pre-trained) models,
preferably convolutional. Supported model frameworks are:

- Tensorflow
- Keras
- Onnx
- Pytorch
- Caffe
- Chainer

It would be convenient if the code used to create the model is also uploaded
to the ```code``` sub-folder due to all the helpful information it can provide.

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
To create a virtual environment,
[download and install](https://docs.conda.io/en/latest/miniconda.html)
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

Once the dependencies are installed, the notebooks in the ```code``` subfolder
can be executed in the environment containing those packages.
