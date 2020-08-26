#include "mlpack_to_torch.hpp"
// #include <torch/torch.h>
#include <string>

torch::nn::Sequential& makeModel()
{
  torch::nn::Sequential model;
  // model->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(84, 4, 1).with_bias(false)));
  model->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3,
      4, {5,
      6}).stride({7,
      8}).padding({(8) / 2, (8 / 2)})));
  model->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(
  {2, 3}).stride({4, 4})));
  model->push_back(torch::nn::Linear(torch::nn::LinearOptions(3, 4)));
  model->push_back(torch::nn::Functional(torch::leaky_relu, 2));
  // model->push_back(torch::nn::Functional(torch::prelu, 1));
  model->push_back(torch::nn::Functional(torch::elu, 1, 1, 1));
  std::cout<< "Model created.";
  return model;
}

void testModel()
{
  FFN<> net;
  net.Add<Linear<>>(4, 10);
  net.Add<ReLULayer<>>();
  net.Add<Linear<>>(10, 1);
  net.Add<ReLULayer<>>();
  // convert(net);

  std::vector<LayerTypes<> > layers = net.Model();
  std::vector<std::string> layerTypes;
  for (LayerTypes<> layer : layers)
  {
    layerTypes.push_back(boost::apply_visitor(LayerNameVisitor(), layer));
  }

  for (auto l : layerTypes)
  {
    std::cout << l << '\n';
  }

  torch::nn::Sequential torchModel;
  auto layerTypeItr = layerTypes.begin();
  auto layerItr = layers.begin();
  for (; layerTypeItr != layerTypes.end(); ++layerTypeItr, ++layerItr)
  {
    auto layerType = *layerTypeItr;
    std::cout << layerType << '\n';
    if (layerType == "linear")
    {
      Linear<>* layer = reinterpret_cast<Linear<>*>(&*layerItr);
      std::unordered_map<std::string, double> values;
      values = boost::apply_visitor(LayerTypeVisitor(), *layerItr);
      std::cout << values["inputsize"] << '\t' << values["outputsize"] << '\n';

      // std::cout << (size_t)layer->InputSize() << '\t' << (size_t)layer->OutputSize() << '\n';
      // torchModel->push_back(torch::nn::Linear(torch::nn::LinearOptions(layer->InputSize(),
      // layer->OutputSize())));
    }
    else if (layerType == "relu")
    {
      torchModel->push_back(torch::nn::Functional(torch::relu));
    }
  }

  Linear<> linear(5, 6);
  LayerTypes<> lin = new Linear<>(5, 6);
  size_t width = boost::apply_visitor(OutputWidthVisitor(), lin);
  Linear<>* layerCopy = reinterpret_cast<Linear<>*>(&lin);
  std::cout << "Linear dims: " << linear.InputSize() << '\t' << linear.OutputSize() << '\n';
  std::cout << "Linear dims: " << layerCopy->InputSize() << '\t' << layerCopy->OutputSize() << '\n';
  std::cout << "Width: " << width << '\n';
}

struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + another_bias;
  }
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};

void finalTest(int mid)
{
  FFN<NegativeLogLikelihood<>, RandomInitialization> net;
  net.Add<Linear<>>(4, mid); // 4, 2
  // net.Add<ReLULayer<>>();
  net.Add<Linear<>>(mid, 4); // 2, 1
  // net.Add<ReLULayer<>>();
  // torch::nn::Sequential torchModel = convert(net);

  std::vector<LayerTypes<> > layers = net.Model();
  std::vector<std::string> layerTypes;
  for (LayerTypes<> layer : layers)
  {
    layerTypes.push_back(boost::apply_visitor(LayerNameVisitor(), layer));
  }
  torch::nn::Sequential torchModel = transferLayers(layerTypes, layers);
  std::cout << "Returned\n";
  // torch::save(torchModel, "torch_linear_model.pt");

  for (const auto& pair : torchModel->named_parameters())
  {
    std::cout << pair.key() << ": " << pair.value() << std::endl;
  }
  std::cout << "The end.\n";
  net.ResetParameters();
  std::cout << net.Parameters().size() << '\n' << torchModel->parameters().size() << '\n';
  net.Parameters().print();
  transferParameters(net.Parameters(), torchModel->parameters());
  std::cout << "Torch params:\n";
  for (const auto& pair : torchModel->named_parameters())
  {
    std::cout << pair.key() << ": " << pair.value() << std::endl;
  }
  arma::mat inputs = arma::ones(4, 1);
  // inputs.print();
  arma::mat predOut1;
  net.Predict(inputs, predOut1);
  std::cout << "Pred out for mlpack model:\n";
  predOut1.print();
  std::cout << "Pred out for torch model:\n";
  auto predOut2 = torchModel->forward(torch::ones({1, 4}));
  std::cout << predOut2 << '\n';
  // torchModel->pretty_print(cout);

  // Net torchNet(4, 5);
  // std::cout << torchNet.forward(torch::ones({1, 4})) << std::endl;
}

template<
  typename eT
>
void copyWeights(arma::mat armaWt, at::Tensor torchWt)
{

}

void convTest()
{
  FFN<NegativeLogLikelihood<>, RandomInitialization> net;
  net.Add<Convolution<>>(3, 4, 2, 3, 1, 1, 0, 0, 64, 128); // 4, 2

  std::vector<LayerTypes<> > layers = net.Model();
  std::vector<std::string> layerTypes;
  for (LayerTypes<> layer : layers)
  {
    layerTypes.push_back(boost::apply_visitor(LayerNameVisitor(), layer));
  }
  torch::nn::Sequential torchModel = transferLayers(layerTypes, layers);

  for (const auto& pair : torchModel->named_parameters())
  {
    std::cout << pair.key() << ": " << pair.value() << std::endl;
  }
  std::cout << "The end.\n";

  for (auto& param : torchModel->parameters())
  {
    param.print();
  }
  std::cout << "\n\nmlpack params:\n";
  net.ResetParameters();
  net.Parameters().print();
  std::cout << "Param len: " << net.Parameters().n_rows;

  arma::cube weight = arma::cube(net.Parameters().memptr(), 2, 3,
        12, false, false);
  weight.print();

  arma::cube weight2 = arma::cube(net.Parameters().memptr(), 3, 2,
        12, false, false); // This method does not work
  weight2.print();

  for (auto& torchParam : torchModel->parameters())
  {

  }

  // [cube slice 0]
  //  0.5736   0.4213  -0.9615
  // -0.4990   0.8933  -0.1902

  // [cube slice 1]
  // -0.4974   0.0413  -0.4516
  // -0.9546  -0.3107   0.1221

  // [cube slice 2]
  // -0.7199   0.0438  -0.0005
  //  0.0877   0.7142  -0.1613

  // arma::cube bias = arma::mat(weights.memptr() + weight.n_elem,
  //       outSize, 1, false, false);
}

void paramTest()
{
  FFN<NegativeLogLikelihood<>, RandomInitialization> net;
  Linear<> l1(4, 2);
  Linear<> l2(2, 1);
  l1.Reset();
  l2.Reset();
  net.Add<Linear<>>(l1);
  net.Add<ReLULayer<>>();
  net.Add<Linear<>>(l2);

  net.ResetParameters();
  cout << "Parameters of model:\n";
  net.Parameters().print();
  arma::mat inputs = arma::ones(4, 1);
  // inputs.print();
  arma::mat predOut1;
  net.Predict(inputs, predOut1);
  std::cout << "Pred out for mlpack model:\n";
  predOut1.print();
  std::cout << "Individual parameters:\n";
  l1.Parameters().print();
  std::cout << '\n';
  l2.Parameters().print();
}

int main()
{
    std::string inFile = "Examples/mlpack_linear_model.xml";
    std::string outFile = "torch_linear_model.pt";
    FFN<> mlpackModel;
    data::Load(inFile, "mlpack_model", mlpackModel);
    // convertModel(inFile, outFile);
    // testModel();
    // makeModel();

    // finalTest(2);
    // cout << "\n\n\n";
    // finalTest(1);

    // paramTest();

    convTest();
}
