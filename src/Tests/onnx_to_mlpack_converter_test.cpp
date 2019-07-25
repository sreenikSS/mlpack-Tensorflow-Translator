#include <mlpack/core.hpp>
#include <boost/test/unit_test.hpp>
#include <mlpack/tests/test_tools.hpp>
#include <mlpack/core/data/split_data.hpp>

// Just copy-pasted some common includes from a previously written program, so not too contextual
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>

// generally just #include <torch/torch.h> for other systems
#include <torch/csrc/api/include/torch/torch.h>
#include <torch/csrc/onnx/onnx.h>
#include <torch/csrc/onnx/init.h>
#include "onnx_to_mlpack.cpp"

#include <ensmallen.hpp>


using namespace mlpack;
using namespace mlpack::ann;

using namespace arma;
using namespace std;

using namespace ens;

class Dataset
{
    private:
        arma::mat armaTrainX, armaTrainY, armaTestX, armaTestY;
    public:
        // There will be other constructors too for initializing the variables
        // and other methods for obtaining them
        Dataset();

};

/*
    Generate an arithmetic progression with first term 'a',
    common difference 'd' and number of terms 'n'.
    If inputs is 100 then a series of 100 training examples will be generated.
 */
int generateAP(int a, int d, int n, int inputs)
{
    // This is one kind of dataset I thought I can easily generate and train upon,
    // even images (random noise basically) can be generated in this format and checked
    // for accuracy as an accuracy of even 10% for the torch model and also 10% for the
    // converted mlpack model would mean that the conversion is correct
    int X[n];

}

/*
    Generate an object of class Dataset using generateAP()
 */
int generateDataset(Dataset& dataset)
{

}

// Where to find the MNIST dataset.
const char* kDataRoot = "./models";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 3;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

struct Net : torch::nn::Module
{
  Net()
      : conv1(torch::nn::Conv2dOptions(1, 10, 5)),
        conv2(torch::nn::Conv2dOptions(10, 20, 5)),
        fc1(320, 50),
        fc2(50, 10) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(
        torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, 0.5, is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::FeatureDropout conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

template <typename DataLoader>
void train(
    int32_t epoch,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size)
{
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader)
  {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0)
    {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
double test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size)
{
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }
  test_loss /= dataset_size; // can use this value also for checking
  return (double) correct / dataset_size;
}

double buildTorchCNNModel()
{
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    //std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    //std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Net model;
  model.to(device);

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) 
  {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
  }
  torch::onnx::export(model, "onnx_mnist_model.onnx");
  return test(model, device, *test_loader, test_dataset_size);
}

/*
    Build similar torch and mlpack models, use the converter to convert the torch model to mlpack
    (torch has a default onnx converter in built) and finally test the accuracy
 */
int trainModel()
{
  // Either this function will create a number of different models or there will be different
  // functions like trainLinearModel(), trainConvolutionalModel(), etc.
  double torchAccuracy = buildTorchCNNModel();
  
  std::ifstream in("onnx_mnist_model.onnx", std::ios_base::binary);
  model.ParseFromIstream(&in);
  in.close();

  GraphProto graph = model.graph();
  storedParams["inputwidth"] = 28; // actually not initializable here, will need to add this as a parameter to one of the functions needing it
  storedParams["inputheight"] = 28;
  FFN<> ffnModel = generateModel(graph);
  extractWeights(graph, ffnModel.Parameters());
  
  double mlpackAccuracy;
  // test the mlpack model against the same test set and obtain the accuracy
  BOOST_REQUIRE_CLOSE(torchAccuracy, mlpackAccuracy, 0.1);
}

// Haven't thought of what all to include here, I was playing with the code here
int main()
{
}
