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
  torch::nn::Sequential torchModel; // change this to a pointer
  auto layerTypeItr = layerTypes.begin();
  auto layerItr = layers.begin();
  for (; layerTypeItr != layerTypes.end(); ++layerTypeItr, ++layerItr)
  {
    std::cout << "Layer start\n";
    auto layerType = *layerTypeItr;
    std::cout << layerType << '\n';
    unordered_map<string, double> values = boost::apply_visitor(LayerTypeVisitor(), *layerItr);
    if (layerType == "linear")
    {
      torchModel->push_back(torch::nn::Linear(torch::nn::LinearOptions(values["insize"], values["outsize"])));
    }
    // else if (layerType == "linearnobias") // getter functions to be implemented
    // {
    //   LinearNoBias<>* layer = reinterpret_cast<LinearNoBias<>*>(&*layerItr);
    //   torchModel->push_back(torch::nn::Linear(layer->InputSize(),
    //   layer->OutputSize(), false));
    // }
    // else if (layerType == "batchnorm")
    // {
    //     // not yet made customizable (eps and size)
    //     // torchModel->push_back(torch::nn::BatchNorm());
    // }
    // else if (layerType == "constant")
    // {
        // to be done later
    // }
    else if (layerType == "convolution")
    {
      torchModel->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(values["insize"],
      values["outsize"], {(int)values["kh"], (int)values["kw"]}).stride({(int)values["dh"],
      (int)values["dw"]}).padding({(int)(values["padht"] +
      values["padhb"]) / 2, (int)(values["padwl"] + values["padwr"]) / 2})));
    }
    else if (layerType == "dropout")
    {
      // is not needed as retraining is not yet supported
    }
    else if (layerType == "leakyrelu")
    {
      torchModel->push_back(torch::nn::Functional(torch::leaky_relu,
      values["alpha"]));
    }
    else if (layerType == "logsoftmax")
    {
      // torchModel->push_back(torch::nn::Functional(torch::log_softmax, 1, torch::nullopt));
    }
    // else if (layerType == "transposedconvolution")
    // {
    //   // need to use Conv2dOptions
    // }
    else if (layerType == "elu")
    {
      // recheck the scale and "input scale" once
      torchModel->push_back(torch::nn::Functional(torch::elu,
      values["alpha"], 1, 1));
    }
    else if (layerType == "maxpooling")
    {
      MaxPooling<>* layer = reinterpret_cast<MaxPooling<>*>(&*layerItr);
      torchModel->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(
      {(int)values["kh"], (int)values["kw"]}).
      stride({(int)values["dh"], (int)values["dw"]})));
      //1, !layer->Floor()));
    }
    else if (layerType == "identity")
    {
      // torchModel->push_back(torch::nn::Sequential()); // not too sure
    }
    else if (layerType == "prelu")
    {
      // Compilation error

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
    // else if (layerType == "softsign")
    // {
    //   // could not find a direct implementation
    // }
    else if (layerType == "tanh")
    {
      torchModel->push_back(torch::nn::Functional(torch::tanh));
    }
    else
    {
      Log::Fatal << "Unsupported layer: " << layerType << '\n';
    }
    std::cout << "Layer end\n";
  }
  std::cout << "Returning\n";
  return torchModel;
}

/**
 * Transfer the weights of the mlpack model to the torch model.
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
    // torchParam = torchParam.view({-1, 1});

    auto origSize = torchParam.sizes();
    torchParam = torchParam.t();
    std::cout << origSize << '\n' << torchParam.sizes() << "\n\n";

    torchParam = torchParam.view({-1, 1});
    
    auto paramAccessor  = torchParam.accessor<float, 2>();

    // std::cout << paramAccessor.sizes() << '\n';
    std::cout << torchParam.sizes() << '\n';

    for (int i = 0; i < paramAccessor.size(0); ++i)
    {
      // for (int j = 0; j < paramAccessor.size(1); ++j)
      // {
      //   paramAccessor[i][j] = mlpackParams(mlpackParamItr++, 0);
      //   std::cout << "i= " << i << " j= " << j << " mlpackParamItr= " << mlpackParamItr << " value= " << paramAccessor[i][j] << '\n';
      // }
      paramAccessor[i][0] = mlpackParams(mlpackParamItr++, 0);
      std::cout << "i= " << i << " mlpackParamItr= " << mlpackParamItr << " value= " << paramAccessor[i][0] << '\n';
    }

    torchParam = torchParam.view(origSize);
  }



  //   int totalParam = 1;
  //   // torchParam = torchParam.view({-1, 1});

  //   auto origSize = torchParam.sizes();

  //   std::cout << origSize << "\n\n";

  //   torchParam = torchParam.view({-1, 1});
    
  //   auto paramAccessor  = torchParam.accessor<float, 2>();

  //   // std::cout << paramAccessor.sizes() << '\n';
  //   std::cout << torchParam.sizes() << '\n';

  //   for (int i = 0; i < paramAccessor.size(0); ++i)
  //   {
  //     // for (int j = 0; j < paramAccessor.size(1); ++j)
  //     // {
  //     //   paramAccessor[i][j] = mlpackParams(mlpackParamItr++, 0);
  //     //   std::cout << "i= " << i << " j= " << j << " mlpackParamItr= " << mlpackParamItr << " value= " << paramAccessor[i][j] << '\n';
  //     // }
  //     paramAccessor[i][0] = mlpackParams(mlpackParamItr++, 0);
  //     std::cout << "i= " << i << " mlpackParamItr= " << mlpackParamItr << " value= " << paramAccessor[i][0] << '\n';
  //   }
  //   if (origSize.size() == 1)
  //     torchParam = torchParam.view(origSize);
  //   else
  //     torchParam = torchParam.reshape({origSize[1], origSize[0]}).t();
  // }
}

/**
 * Get a torch model with the layer types, details
 * and weights corresponding to the given mlpack model.
 * 
 * @param mlpackModel The mlpack model that is to be converted.
 * @return A sequential torch model corresponding to the given
 * mlpack model with the weights also transferred.
 */
torch::nn::Sequential& convert(FFN<>& mlpackModel) // remove reference
{
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