#include <onnx/onnx_pb.h>
//#include <onnx/proto_utils.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "model_parser.hpp"

using namespace onnx;
using namespace mlpack;
using namespace mlpack::ann;
using namespace std;

map<string, double> storedParams;

LayerTypes<> getLayer(const NodeProto& node, string layerType, map<string, double>& dimParams)
{
    map<string, double> mappedParams;

    map<string, string> operatorMap;
    // keys are onnx operators, values are the corresponding mlpack layer names
    operatorMap = {
        {"BatchNormalization", "batchnorm"},
        {"ConstantOfShape", "constant"},// probably this and not Constant
        {"Conv", "convolution"},
        {"Dropout", "dropout"},
        {"LeakyRelu", "leakyrelu"},
        {"Transformed_linear", "linear"},
        {"Gemm", "linear"},// not Gemm, but Matmul (ambiguous stuff)
        {"MatMul", "linearnobias"},
        {"LogSoftmax", "logsoftmax"},
        {"ConvTranspose", "transposedconvolution"},
        {"Elu", "elu"},
        {"MaxPool", "maxpooling"},
        {"Identity", "identity"},
        {"PRelu", "prelu"},
        {"Relu", "relu"},
        {"Selu", "selu"},
        {"Sigmoid", "sigmoid"},
        {"Softmax", "logsoftmax"},// Needs implementation in mlpack, temporary fix to logsoftmax for testing
        {"Softsign", "softsign"},// Needs to be defined in the parser
        {"Tanh", "tanh"}
    };

    //map< map<string, string>, vector<string>> mappedNames;
    map<string, map<string, vector<string>>> mappedNames;


    // Missing support for reshape layer (but I guess it is done automatically here)

    //mappedNames["BatchNormalization"];
    mappedNames["Conv"] = {
        {"kernel_shape", {"kh", "kw"}},
        {"pads", {"padh", "padw"}},
        {"strides", {"dh", "dw"}}  
    };

    mappedNames["Dropout"] = {
        {"ratio", {"ratio"}}
    };

    mappedNames["LeakyRelu"] = {
        {"alpha", {"alpha"}}
    };

    mappedNames["Gemm"];

    mappedNames["MatMul"];

    mappedNames["LogSoftmax"];

    mappedNames["ConvTranspose"] = {
        {"kernel_shape", {"kh", "kw"}},
        {"pads", {"padh", "padw"}},
        {"strides", {"dh", "dw"}}  
    };

    mappedNames["Elu"] = {
        {"alpha", {"alpha"}}
    };

    mappedNames["MaxPool"] = {
        {"kernel_shape", {"kh", "kw"}},
        {"strides", {"dh", "dw"}}  
    };// support for 'pads' missing in mlpack

    mappedNames["Identity"];

    mappedNames["PRelu"] = {
        {"slope", {"alpha"}}
    };// 'slope' is actually not an attribute rather an input

    mappedNames["Relu"];

    mappedNames["Selu"];// support for custom alpha or gamma missing in mlpack

    mappedNames["Sigmoid"];

    mappedNames["Softmax"];// not yet implemented in mlpack

    mappedNames["Softsign"];// not yet implemented in the parser

    mappedNames["Tanh"]; 

    map<string, vector<string>> layer = mappedNames[layerType];
    // stores the attributes which can be calculated only after knowing the other ones
    vector<string> skippedAttributes;

    for (AttributeProto attribute:node.attribute())
    {
        string attrName = attribute.name();
        //std::cout << attribute.name() << "\n";
        vector<string> attr = layer[attrName];
        vector<string>::iterator itr;

        //check for special cases
        if (attrName == "pads")
        {
            // [0 1 2 3] indices are top, bottom, left, right respectively
            mappedParams["padw"] = (int) (attribute.ints()[1] + attribute.ints()[3]) / 2;
            mappedParams["padh"] = (int) (attribute.ints()[0] + attribute.ints()[2]) / 2;
            continue;
        }
        else if (attrName == "auto_pad")
        {
            
            // P = ((S-1)*W-S+F)/2
            skippedAttributes.push_back("auto_pad_" + attribute.s());
        }
        int i = 0;
        // validation needs to be added
        for (itr = attr.begin(); itr < attr.end(); ++itr, ++i)
        {
            if (attribute.type() == attribute.INT)
            {
                mappedParams[*itr] = attribute.i();
            }
            else if (attribute.type() == attribute.INTS)
            {
                mappedParams[*itr] = attribute.ints()[i];
            }
            else if (attribute.type() == attribute.FLOAT)
            {
                mappedParams[*itr] = attribute.f();
            }
            else if (attribute.type() == attribute.FLOATS)
            {
                mappedParams[*itr] = attribute.floats()[i];
            }
        }        
    }
    
    map<string, double>::iterator itr;
    for (itr = dimParams.begin(); itr != dimParams.end(); ++itr)
    {
        mappedParams[itr->first] = itr->second;
    }

    for (string& attribute:skippedAttributes)
    {
        if (attribute ==  "auto_pad_SAME")
        {
            mappedParams["padw"] = (int) ((mappedParams["inputwidth"] * (mappedParams["dw"] - 1) - mappedParams["dw"] + mappedParams["kw"] + 1) / 2);
            mappedParams["padh"] = (int) ((mappedParams["inputheight"] * (mappedParams["dh"] - 1) - mappedParams["dh"] + mappedParams["kh"] + 1) / 2);
        }
        else if (attribute == "auto_pad_VALID") // not absolutely necessary though as the default is zero pad
        {
            mappedParams["padw"] = 0;
            mappedParams["padh"] = 0;
        }
    }

    // store dimensional details for the next layer if needed
    if(layerType == "Conv")
    {
        // out_w = ((in_w + 2 * (pad_w) - kernel_w - (kernel_w - 1) * (dilation_w - 1))/stride_w) + 1;
		// 	out_h = ((in_h + 2 * (pad_h) - kernel_h - (kernel_h - 1) * (dilation_h - 1))/stride_h) + 1;
        // Convolution<> layer(mappedParams["insize"], mappedParams["outsize"],
        // mappedParams["kw"], mappedParams["kh"], mappedParams["dw"],
        // mappedParams["dh"], mappedParams["padw"], mappedParams["padh"],
        // mappedParams["inputwidth"], mappedParams["inputheight"]);

        // storedParams["inputwidth"] = layer.OutputWidth(); 
        // storedParams["inputheight"] = layer.OutputHeight();
        storedParams["inputwidth"] = (mappedParams["inputwidth"] - mappedParams["kw"] + 2 * mappedParams["padw"]) / mappedParams["dw"] + 1;
        storedParams["inputheight"] = (mappedParams["inputheight"] - mappedParams["kh"] + 2 * mappedParams["padh"]) / mappedParams["dh"] + 1;
    }

    cout << "Layer type of mlpack model: " << operatorMap[layerType] << "\nLayer map:\n";
    printMap(mappedParams);
    cout << "\n\n";
    return getNetworkReference(operatorMap[layerType], mappedParams);

}

bool isBiasFirst(auto weights)
{

}

std::vector<int> findWeightDims(auto& weights) // a real waste of compute, will be done away with soon
{
    //cout << "In findWeightDims()\n";
    std::vector<int> dims;
    auto itr = std::begin(weights);
    if ((*itr).dims().size() > 2)
    {
        //cout << "Dim: " << (*itr).dims(1) << "\n";
        dims.push_back((*itr).dims(1));
        //cout << "Dim: " << (*itr).dims(0) << "\n";
        dims.push_back((*itr).dims(0));
        itr += 2;
    }
    for (; itr != std::end(weights); itr += 2)
    {
        if ((*itr).dims().size() == 0)
        {
            itr--;
        }
        else
        {
            //cout << "Dim: " << (*itr).dims(0) << "\n";
            dims.push_back((*itr).dims(0));
        }
    }
    //cout << "Most dims pushed back\n";
    itr -= 2;
    dims.push_back((*itr).dims(1));
    return dims;
}

// Currently redundant
std::vector<int> findWeightDims2(auto& weights)
{
    cout << "In findWeightDims()\n";
    std::vector<int> dims;
    auto itr = std::begin(weights);
    for (; itr != std::end(weights); itr += 2)
    {
        if ((*itr).dims().size() == 0)
        {
            itr--;
        }
        else if ((*itr).dims().size() > 2)
        {
            cout << "Dim: " << (*itr).dims(1) << "\n";
            dims.push_back((*itr).dims(1));
        }
        else
        {
            cout << "Dim: " << (*itr).dims(0) << "\n";
            dims.push_back((*itr).dims(0));
        }
    }
    cout << "Most dims pushed back\n";
    itr -= 2;
    dims.push_back((*itr).dims(1));
    return dims;
}

void extractWeights(GraphProto& graph, arma::mat& weightMatrix)
{
    auto& weights = graph.initializer();
    int totalWeights = 0;
    int weightDims[weights.size()];
    int wtNumber = 0;
    for(TensorProto weight:weights)
    {
        if (weight.dims().size() == 0)
            continue;
        int weightSize = weight.dims(0);
        for (int i = 1; i < weight.dims_size(); ++i)
        {
            weightSize *= weight.dims(i);
        }
        weightDims[wtNumber++] = weightSize;
        totalWeights += weightSize;
    }
    wtNumber = 0;// reinitialize for use
    int count = 0;
    weightMatrix.set_size(totalWeights, 1);
    //cout << "Total weights: " << totalWeights << "\n";
    for(TensorProto weight:weights)
    {
        if (weight.dims().size() == 0)
            continue; 
        //cout << "\nWeight type: " << weight.data_type() << " Weight size: " << weightDims[wtNumber] << "\n";
        if (weight.has_raw_data())
        {
            std::string rawData = weight.raw_data();
            const char* ws = rawData.c_str();
            for (int i = 0; i < weightDims[wtNumber] * 4 - 4; i += 4)
            {
                float wt;
                char t[] = {ws[i], ws[i+1], ws[i+2], ws[i+3]};
                memcpy(&wt, &t, sizeof(float));
                weightMatrix(count++, 0) = wt;
            }
        }
        else if (weight.data_type() == 1)// for float type
        {
            for (int i = 0; i < weightDims[wtNumber]; ++i)
            {
                weightMatrix(count++, 0) = weight.float_data()[i];
                //cout << "\nWeight " << count - 1 << " : " << weightMatrix(count - 1, 0);
            }
        }
        wtNumber++;
    }
}

FFN<> generateModel(GraphProto& graph)
{
    FFN<> mod;
    std::vector<int> dims = findWeightDims(graph.initializer());
    std::vector<int>::iterator itr = dims.begin();
    //cout << "Before traversing nodes\n";
    for (auto nodeItr = std::begin(graph.node()); nodeItr != std::end(graph.node()); ++nodeItr)
    {
        string nodeType = nodeItr -> op_type();
        map<string, double> dimParams;
        std::vector<string> skipLayers = {"Add", "Identity", "Reshape", "Transpose", "Unsqueeze",
        "Shape", "Cast", "Slice", "Concat", "ReduceProd"};
        std::vector<string> dimensionalLayers = {"ConstantOfShape", "Conv", "Gemm", "Matmul",
        "ConvTranspose", "Transformed_linear"};// may be an error with the ConstantOfShape layer
        // layers for merging, currently just one is there
        std::map<std::vector<string>, string> mergeLayers = {
            {{"MatMul", "Add"}, "Transformed_linear"}
            };

        std::map<std::vector<string>, string>::iterator mergeItr;
        for (mergeItr = mergeLayers.begin(); mergeItr != mergeLayers.end(); ++mergeItr)
        {
            std::vector<string> mergeVector = mergeItr -> first;
            int i;
            for (i = 0; i < mergeVector.size(); ++i)
            {
                //cout << "i= " << i << " mervector[i]= " << mergeVector[i] << " op_type= " << (nodeItr + i) -> op_type() << "\n";
                if ((nodeItr + i) == std::end(graph.node()) || mergeVector[i] != (nodeItr + i) -> op_type())
                    break;
            }
            //cout << "i= " << i << " mergeVector.size()= " << mergeVector.size() << "\n";
            if (i == mergeVector.size())
            {
                nodeItr += i;
                nodeType = mergeItr -> second;
            }
        }

        if (std::find(skipLayers.begin(), skipLayers.end(), nodeType) != skipLayers.end())
            continue;
        
        if (std::find(dimensionalLayers.begin(), dimensionalLayers.end(), nodeType) != dimensionalLayers.end())
        {
            dimParams["insize"] = *itr;
            itr++;
            dimParams["outsize"] = *itr;
        }
        if(nodeType == "Conv")
        {
            dimParams["inputwidth"] = storedParams["inputwidth"];
            dimParams["inputheight"] = storedParams["inputheight"];
        }
        std::cout << "Node type: " << nodeType << "\n";
        printMap(dimParams);
        mod.Add(getLayer(*nodeItr, nodeType, dimParams));
    }
    return mod;
}

int main()
{
    ModelProto model;
    // Todo:
    // check the input and output width and height of image for convolution (done)
    // add same and valid padding support (done)
    // check input size (channel no) of first conv layer (done)
    // check if linearnobias creates an empty bias array in the onnx model or no bias array at all
    std::ifstream in("onnx_conv_model.onnx", std::ios_base::binary);
    //std::ifstream in("onnx_linear_model.onnx", std::ios_base::binary);
    model.ParseFromIstream(&in);
    in.close();

    GraphProto graph = model.graph();
    storedParams["inputwidth"] = 28;
    storedParams["inputheight"] = 28;
    //cout << "Still in main\n";
    FFN<> ffnModel = generateModel(graph);
    //arma::mat temp;
    extractWeights(graph, ffnModel.Parameters());
    //temp.print();
    ffnModel.Parameters().print();
    data::Save("mlpack_conv_model.xml", "conv_model", ffnModel);
    //data::Save("mlpack_linear_model.xml", "linear_model", ffnModel);
    std::cout<<model.graph().node().size()<<"\n";
    std::cout << "Dims of weights: " << graph.initializer().size() << "\n";
}