/**
 * @file layer_details.hpp
 * @author Sreenik Seal
 *
 * Implementation of a class that converts a given ann layer to string format.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <boost/variant/static_visitor.hpp>
#include <unordered_map>
#include <string>

using namespace mlpack::ann;

/**
 * Implementation of a class that returns the string representation of the
 * name of the given layer.
 */
class LayerTypeVisitor :
    public boost::static_visitor<std::unordered_map<std::string, double> >
{
 public:
  //! Create the LayerNameVisitor object.
  LayerTypeVisitor()
  {
    // Nothing to do here.
  }


  /*
   * Return the name of the given layer of type Linear as a map.
   *
   * @param Given layer of type Linear.
   * @return The string representation of the layer.
   */
  std::unordered_map<std::string, double> LayerString(
      AtrousConvolution<>* layer) const
  {
    std::unordered_map<std::string, double> values;
    values["insize"] = layer->InputSize();
    values["outsize"] = layer->OutputSize();
    values["kw"] = layer->KernelWidth();
    values["kh"] = layer->KernelHeight();
    values["dw"] = layer->StrideWidth();
    values["dh"] = layer->StrideHeight();
    values["padwl"] = layer->Padding().PadWLeft();
    values["padwr"] = layer->Padding().PadWRight();
    values["padht"] = layer->Padding().PadHTop();
    values["padhb"] = layer->Padding().PadHBottom();
    values["inputwidth"] = layer->InputWidth();
    values["inputheight"] = layer->InputHeight();
    values["dilationw"] = layer->DilationWidth();
    values["dilationh"] = layer->DilationHeight();
    return values;
  }

  /*
   * Return the name of the given layer of type Linear as a map.
   *
   * @param Given layer of type Linear.
   * @return The string representation of the layer.
   */
  std::unordered_map<std::string, double> LayerString(
      AlphaDropout<>* layer) const
  {
    std::unordered_map<std::string, double> values;
    values["ratio"] = layer->Ratio();
    values["alphadash"] = layer->AlphaDash();
    return values;
  }

  /*
   * Return the name of the given layer of type Linear as a map.
   *
   * @param Given layer of type Linear.
   * @return The string representation of the layer.
   */
  std::unordered_map<std::string, double> LayerString(BatchNorm<>* layer) const
  {
    std::unordered_map<std::string, double> values;
    values["size"] = layer->InputSize();
    values["eps"] = layer->Epsilon();

    // Include commented parts after next mlpack release.
    //values["average"] = layer->Average();
    //values["momentum"] = layer->Momentum();
    return values;
  }

  /*
   * Return the name of the given layer of type Linear as a map.
   *
   * @param Given layer of type Linear.
   * @return The string representation of the layer.
   */
  std::unordered_map<std::string, double> LayerString(Constant<>* layer) const
  {
    std::unordered_map<std::string, double> values;
    values["outsize"] = layer->OutSize();
    return values;
  }

  /*
   * Return the name of the given layer of type Linear as a map.
   *
   * @param Given layer of type Linear.
   * @return The string representation of the layer.
   */
  std::unordered_map<std::string, double> LayerString(
      Convolution<>* layer) const
  {
    std::unordered_map<std::string, double> values;
    values["insize"] = layer->InputSize();
    values["outsize"] = layer->OutputSize();
    values["kw"] = layer->KernelWidth();
    values["kh"] = layer->KernelHeight();
    values["dw"] = layer->StrideWidth();
    values["dh"] = layer->StrideHeight();
    values["padwl"] = layer->PadWLeft();
    values["padwr"] = layer->PadWRight();
    values["padht"] = layer->PadHTop();
    values["padhb"] = layer->PadHBottom();
    values["inputwidth"] = layer->InputWidth();
    values["inputheight"] = layer->InputHeight();
    return values;
  }

  /*
   * Return the name of the given layer of type Linear as a map.
   *
   * @param Given layer of type Linear.
   * @return The string representation of the layer.
   */
  std::unordered_map<std::string, double> LayerString(ELU<>* layer) const
  {
    std::unordered_map<std::string, double> values;
    values["alpha"] = layer->Alpha();
    values["lambda"] = layer->Lambda();
    return values;
  }

  /*
   * Return the name of the given layer of type Linear as a map.
   *
   * @param Given layer of type Linear.
   * @return The string representation of the layer.
   */
  std::unordered_map<std::string, double> LayerString(Linear<>* layer) const
  {
    std::unordered_map<std::string, double> values;
    values["insize"] = layer->InputSize();
    values["outsize"] = layer->OutputSize();
    return values;
  }

  /*
   * Return the name of the given layer of type Linear as a map.
   *
   * @param Given layer of type Linear.
   * @return The string representation of the layer.
   */
  std::unordered_map<std::string, double> LayerString(
      LinearNoBias<>* layer) const
  {
    std::unordered_map<std::string, double> values;
    values["insize"] = layer->InputSize();
    values["outsize"] = layer->OutputSize();
    return values;
  }

  /*
   * Return the name of the given layer of type Linear as a map.
   *
   * @param Given layer of type Linear.
   * @return The string representation of the layer.
   */
  std::unordered_map<std::string, double> LayerString(
      MaxPooling<>* layer) const
  {
    std::unordered_map<std::string, double> values;
    values["insize"] = layer->InputSize();
    values["outsize"] = layer->OutputSize();
    values["kw"] = layer->KernelWidth();
    values["kh"] = layer->KernelHeight();
    values["dw"] = layer->StrideWidth();
    values["dh"] = layer->StrideHeight();
    values["floor"] = layer->Floor();
    values["inputwidth"] = layer->InputWidth();
    values["inputheight"] = layer->InputHeight();
    return values;
  }

  /*
   * Return the name of the given layer of type Linear as a map.
   *
   * @param Given layer of type Linear.
   * @return The string representation of the layer.
   */
  std::unordered_map<std::string, double> LayerString(
      MeanPooling<>* layer) const
  {
    std::unordered_map<std::string, double> values;
    values["insize"] = layer->InputSize();
    values["outsize"] = layer->OutputSize();
    values["kw"] = layer->KernelWidth();
    values["kh"] = layer->KernelHeight();
    values["dw"] = layer->StrideWidth();
    values["dh"] = layer->StrideHeight();
    values["floor"] = layer->Floor();
    values["inputwidth"] = layer->InputWidth();
    values["inputheight"] = layer->InputHeight();
    return values;
  }

  /*
   * Return the name of the given layer of type Linear as a map.
   *
   * @param Given layer of type Linear.
   * @return The string representation of the layer.
   */
  std::unordered_map<std::string, double> LayerString(
      LeakyReLU<>* layer) const
  {
    std::unordered_map<std::string, double> values;
    values["alpha"] = layer->Alpha();
    return values;
  }

  /*
   * Return the name of the given layer of type Linear as a map.
   *
   * @param Given layer of type Linear.
   * @return The string representation of the layer.
   */
  std::unordered_map<std::string, double> LayerString(
      TransposedConvolution<>* layer) const
  {
    std::unordered_map<std::string, double> values;
    values["insize"] = layer->InputSize();
    values["outsize"] = layer->OutputSize();
    values["kw"] = layer->KernelWidth();
    values["kh"] = layer->KernelHeight();
    values["dw"] = layer->StrideWidth();
    values["dh"] = layer->StrideHeight();
    values["padwl"] = layer->PadWLeft();
    values["padwr"] = layer->PadWRight();
    values["padht"] = layer->PadHTop();
    values["padhb"] = layer->PadHBottom();
    values["inputwidth"] = layer->InputWidth();
    values["inputheight"] = layer->InputHeight();
    return values;
  }

  /*
   * Return the name of the layer of specified type as a string.
   *
   * @param Given layer of any type.
   * @return A string declaring that the layer is unsupported.
   */
  template<typename T>
  std::unordered_map<std::string, double> LayerString(T* /*layer*/) const
  {
    std::unordered_map<std::string, double> values;
    return values;
  }

  //! Overload function call.
  std::unordered_map<std::string, double> operator()(MoreTypes layer) const
  {
    return layer.apply_visitor(*this);
  }

  //! Overload function call.
  template<typename LayerType>
  std::unordered_map<std::string, double> operator()(LayerType* layer) const
  {
    return LayerString(layer);
  }
};
