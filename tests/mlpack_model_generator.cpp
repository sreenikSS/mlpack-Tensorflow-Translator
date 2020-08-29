#include "mlpack_to_torch.hpp"
#include <string>

int main()
{
  // Define the network.
  FFN<NegativeLogLikelihood<>, RandomInitialization> net;
  net.Add<Convolution<>>(3, 4, 3, 3, 1, 1, 0, 0, 8, 8);
  net.Add<LeakyReLU<>>();
  net.Add<MaxPooling<>>(2, 2, 2, 2, true);
  net.Add<Linear<>>(4 * 3 * 3, 2);
  net.Add<LogSoftMax<>>();
  net.ResetParameters();

  // Create the xml file.
  data::Save("tests/mlpack_xml_models/simple_conv_model.xml",
      "mlpack_model_converted_from_onnx", net);

  // Print the layers defined above.
  auto torch_model = convert(net);

  return 0;
}
