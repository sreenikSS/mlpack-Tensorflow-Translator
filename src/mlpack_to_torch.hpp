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
#include <ensmallen.hpp>

#include <torch/torch.h>

using namespace mlpack;
using namespace ann;

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
torch::nn::Sequential& transferLayers(std::vector<std::string>& layerTypes,
                                     std::vector<LayerTypes<>>& layers)
{
  torch::nn::Sequential model;
  auto layerTypeItr = layerTypes.begin();
  auto layerItr = layers.begin();
  for (; layerTypeItr != layerTypes.end(); ++layerTypeItr, ++ layerItr)
  {
    auto layerType = *layerTypeItr;
    if (layerType == "linear")
    {
      Linear<>* layer = reinterpret_cast<Linear<>*>(&*layerItr);
      model->push_back(torch::nn::Linear(layer->InputSize(),
      layer->OutputSize(), true));
    }
    else if (layerType == "linearnobias")
    {
      LinearNoBias<>* layer = reinterpret_cast<LinearNoBias<>*>(&*layerItr);
      model->push_back(torch::nn::Linear(layer->InputSize(),
      layer->OutputSize(), false));
    }
    // else if (layerType == "batchnorm")
    // {
    //     // not yet made customizable (eps and size)
    //     // model->push_back(torch::nn::BatchNorm());
    // }
    // else if (layerType == "constant")
    // {
        // to be done later
    // }
    else if (layerType == "convolution")
    {
      Convolution<>* layer = reinterpret_cast<Convolution<>*>(&*layerItr);
      model->push_back(torch::nn::Conv2d(layer->InputSize(),
      layer->OutputSize(), layer->KernelDims(), layer->StrideDims(),
      layer->PaddingDims(), 1, 1, true, 'zeros'));
    }
    else if (layerType == "dropout")
    {
      // is not needed as retraining is not yet supported
    }
    else if (layerType == "leakyrelu")
    {
      LeakyReLU<>* layer = reinterpret_cast<LeakyReLU<>*>(&*layerItr);
      model->push_back(torch::nn::Functional(torch::leaky_relu,
      layer->Alpha()));
    }
    else if (layerType == "logsoftmax")
    {
      model->push_back(torch::nn::Functional(torch::log_softmax));
    }
    // else if (layerType == "transposedconvolution")
    // {
    //   // need to use Conv2dOptions
    // }
    else if (layerType == "elu")
    {
      ELU<>* layer = reinterpret_cast<ELU<>*>(&*layerItr);
      model->push_back(torch::nn::Functional(torch::elu,
      layer->Alpha()));
    }
    else if (layerType == "maxpooling")
    {
      MaxPooling<>* layer = reinterpret_cast<MaxPooling<>*>(&*layerItr);
      model->push_back(torch::nn::Functional(torch::max_pool2d,
      layer->KernelDims(), layer->StrideDims(), layer->PaddingDims(),
      1, !layer->Floor()));
    }
    else if (layerType == "identity")
    {
      model->push_back(torch::nn::Sequential()); // not too sure
    }
    else if (layerType == "prelu")
    {
      PReLU<>* layer = reinterpret_cast<PReLU<>*>(&*layerItr);
      model->push_back(torch::nn::Functional(torch::prelu, 1,
      layer->Alpha())); // for single channel
    }
    else if (layerType == "relu")
    {
      model->push_back(torch::nn::Functional(torch::relu));
    }
    else if (layerType == "selu")
    {
      model->push_back(torch::nn::Functional(torch::selu));
    }
    else if (layerType == "sigmoid")
    {
      model->push_back(torch::nn::Functional(torch::sigmoid));
    }
    // else if (layerType == "softsign")
    // {
    //   // could not find a direct implementation
    // }
    else if (layerType == "tanh")
    {
      model->push_back(torch::nn::Functional(torch::tanh));
    }
    else
    {
      Log::Fatal << "Unsupported layer: " << layerType << '\n';
    }
  }
  return model;
}

/**
 * Transfer the weights of the mlpack model to the torch model
 * 
 * @param mlpackParams Weights of the mlpack model
 * @param torchParams Weiights of the torch model
 */
void transferParameters(arma::mat& mlpackParams, std::vector<
                        at::Tensor> torchParams)
{
  int mlpackParamItr = 0;
  for (auto& torchParam : torchParams)
  {
    int totalParam = 1;
    torchParam = torchParam.view({-1, 1});
    auto origSize = torchParam.sizes();
    auto paramAccessor  = torchParam.accessor<float, 2>();
    for (int i = 0; i < paramAccessor.size(0); ++i)
    {
      paramAccessor[i][0] = mlpackParams(mlpackParamItr++, 0);
    }
    torchParam = torchParam.view(origSize);
  }
}

/**
 * Get a torch model with the layer types, details
 * and weights corresponding to the given mlpack model
 * 
 * @param mlpackModel The mlpack model that is to be converted
 * @return A sequential torch model corresponding to the given
 * mlpack model with the weights also transferred
 */
torch::nn::Sequential& convert(FFN<>& mlpackModel)
{
  std::vector<LayerTypes<> > layers = mlpackModel.Network();
  std::vector<std::string> layerTypes;
  for (LayerTypes<> layer : layers)
  {
    layerTypes.push_back(boost::apply_visitor(TestVisitor(), layer));
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