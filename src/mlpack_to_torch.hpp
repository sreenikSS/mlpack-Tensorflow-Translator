#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/init_rules/kathirvalavakumar_subavathi_init.hpp>
#include <mlpack/methods/ann/init_rules/lecun_normal_init.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/init_rules/oivs_init.hpp>
#include <mlpack/methods/ann/init_rules/orthogonal_init.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/earth_mover_distance.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <mlpack/methods/ann/loss_functions/kl_divergence.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/loss_functions/reconstruction_loss.hpp>
#include <mlpack/methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp>
#include <mlpack/methods/ann/activation_functions/softsign_function.hpp>
#include <mlpack/methods/ann/activation_functions/swish_function.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer_names.hpp>
#include "layer_details.hpp"
#include <ensmallen.hpp>

#include <torch/torch.h>

using namespace mlpack;
using namespace ann;
using namespace std;

/**
 * Get a torch model with the given layer types and details corresponding to
 * the mlpack model
 *
 * @param layerTypes The vector containing all the layer names
 * @param layers The vector containing the LayerTypes<> object
 * associated with each layer of the mlpack model
 * @return A sequential torch model corresponding to the given
 * mlpack model details
 */
torch::nn::Sequential transferLayers(std::vector<std::string>& layerTypes,
                                     std::vector<LayerTypes<>>& layers)
{
  // #TODO: [IMPORTANT] change this to a pointer
  torch::nn::Sequential torchModel;
  auto layerTypeItr = layerTypes.begin();
  auto layerItr = layers.begin();
  for (; layerTypeItr != layerTypes.end(); ++layerTypeItr, ++layerItr)
  {
    auto layerType = *layerTypeItr;
    // # TODO: Replace cout with Log?
    std::cout << layerType << '\n';
    unordered_map<string, double> values = boost::apply_visitor(
        LayerTypeVisitor(), *layerItr);
    if (layerType == "linear")
    {
      torchModel->push_back(torch::nn::Linear(torch::nn::LinearOptions(
          values["insize"], values["outsize"])));
    }
    /*
    else if (layerType == "linearnobias")
    {
      // #TODO : getter functions to be implemented
      //LinearNoBias<>* layer = reinterpret_cast<LinearNoBias<>*>(&*layerItr);
      //torchModel->push_back(torch::nn::Linear(layer->InputSize(),
      //layer->OutputSize(), false));
    }
    else if (layerType == "batchnorm")
    {
      // #TODO :
      // not yet made customizable (eps and size)
      // torchModel->push_back(torch::nn::BatchNorm());
    }
    else if (layerType == "constant")
    {
        // #TODO :
    }
    */
    else if (layerType == "convolution")
    {
      torchModel->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(
          values["insize"], values["outsize"], {(int)values["kh"],
          (int)values["kw"]}).stride({(int)values["dh"],
          (int)values["dw"]}).padding({(int)(values["padht"] +
          values["padhb"]) / 2, (int)(values["padwl"] +
          values["padwr"]) / 2})));
    }
    else if (layerType == "dropout")
    {
      // #TODO :
      // is not needed as retraining is not yet supported
    }
    else if (layerType == "leakyrelu")
    {
      torchModel->push_back(torch::nn::Functional(torch::leaky_relu,
      values["alpha"]));
    }
    else if (layerType == "logsoftmax")
    {
      // #TODO :
      //torchModel->push_back(torch::nn::Functional(torch::log_softmax, 1,
      //torch::nullopt));

    }
    /*
    else if (layerType == "transposedconvolution")
    {
      // #TODO : need to use Conv2dOptions
      // need to use Conv2dOptions
    }
    */
    else if (layerType == "elu")
    {
      // #TODO : recheck the scale and "input scale".
      torchModel->push_back(torch::nn::Functional(torch::elu,
      values["alpha"], 1, 1));
    }
    else if (layerType == "maxpooling")
    {
      MaxPooling<>* layer = reinterpret_cast<MaxPooling<>*>(&*layerItr);
      torchModel->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(
      {(int)values["kh"], (int)values["kw"]}).
      stride({(int)values["dh"], (int)values["dw"]})));
    }
    else if (layerType == "identity")
    {
      // #TODO :
      // torchModel->push_back(torch::nn::Sequential()); // not too sure
    }
    else if (layerType == "prelu")
    {
      // #TODO : Resolve compilation error

      // PReLU<>* layer = reinterpret_cast<PReLU<>*>(&*layerItr);
      // torchModel->push_back(torch::nn::Functional(torch::prelu, 1,
      // layer->Alpha())); // for single channel
    }
    else if (layerType == "relu")
    {
      torchModel->push_back(torch::nn::Functional(torch::relu));
    }
    else if (layerType == "selu")
    {
      torchModel->push_back(torch::nn::Functional(torch::selu));
    }
    else if (layerType == "sigmoid")
    {
      torchModel->push_back(torch::nn::Functional(torch::sigmoid));
    }
    /*
    else if (layerType == "softsign")
    {
      // #TODO : could not find a direct implementation
    }
    */
    else if (layerType == "tanh")
    {
      torchModel->push_back(torch::nn::Functional(torch::tanh));
    }
    else
    {
      Log::Fatal << "Unsupported layer: " << layerType << '\n';
    }
  }
  return torchModel;
}

/**
 * Transfer the weights of the mlpack model to the torch model.
 *
 * @param mlpackParams Weights of the mlpack model
 * @param torchParams Weights of the torch model
 */
void transferParameters(arma::mat& mlpackParams, std::vector<
                        at::Tensor> torchParams)
{
  int mlpackParamItr = 0;
  for (auto& torchParam : torchParams)
  {
    int totalParam = 1;
    auto origSize = torchParam.sizes();
    int dimLen = torchParam.sizes().size();
    if (dimLen == 1)
    {
      for (int i = 0; i < torchParam.size(0); ++i)
        torchParam.index_put_({i}, mlpackParams(mlpackParamItr++, 0));
    }
    else if (dimLen == 2)
    {
      torchParam = torchParam.t();
      for (int i = 0; i < torchParam.size(0); ++i)
      {
        for (int j = 0; j < torchParam.size(1); ++j)
        {
          torchParam.index_put_({i, j}, mlpackParams(mlpackParamItr++, 0));
        }
      }
    }
    else if (dimLen == 4)
    {
      for (int i = 0; i < torchParam.size(0); ++i)
      {
        for (int j = 0; j < torchParam.size(1); ++j)
        {
          for (int k = 0; k < torchParam.size(3); ++k)
          {
            for (int l = 0; l < torchParam.size(2); ++l)
            {
              torchParam.index_put_({i, j, l, k},
                  mlpackParams(mlpackParamItr++, 0));
            }
          }
        }
      }
    }
  }
}

/**
 * Get a torch model with the layer types, details
 * and weights corresponding to the given mlpack model.
 *
 * @param mlpackModel The mlpack model that is to be converted.
 * @return A sequential torch model corresponding to the given
 * mlpack model with the weights also transferred.
 */
torch::nn::Sequential& convert(FFN<>& mlpackModel)
{
  // # TODO : [IMPORTANT] remove reference (for the parameter mlpackModel)
  std::vector<LayerTypes<> > layers = mlpackModel.Model();
  std::vector<std::string> layerTypes;
  for (LayerTypes<> layer : layers)
  {
    layerTypes.push_back(boost::apply_visitor(LayerNameVisitor(), layer));
  }
  torch::nn::Sequential torchModel = transferLayers(layerTypes, layers);
  transferParameters(mlpackModel.Parameters(), torchModel->parameters());
  return torchModel;
}

/**
 * Save the converted torch model to the given location
 *
 * @param inFilename The path to the mlpack model
 * @param outFileName The path to the torch model
 */
void convertModel(std::string inFileName, std::string outFileName)
{
  FFN<> mlpackModel;
  data::Load(inFileName, "mlpack_model", mlpackModel);
  torch::nn::Sequential torchModel = convert(mlpackModel);
  torch::save(torchModel, outFileName);
}
