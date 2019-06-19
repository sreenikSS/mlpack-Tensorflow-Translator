#include <onnx/onnx_pb.h>
//#include <onnx/proto_utils.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "modelParser.cpp"

using namespace onnx;
using namespace mlpack;
using namespace mlpack::ann;

LayerTypes<> getLayer(NodeProto& node, map<string, double>& dimParams)
{
    string layerType = node.op_type();
    map<string, double> mappedParams;

    map<string, string> operatorMap;
    // keys are onnx operators, values are the corresponding mlpack layer names
    operatorMap = {
        {"BatchNormalization", "batchnorm"},
        {"ConstantOfShape", "constant"},// probably this and not Constant
        {"Conv", "convolution"},
        {"Dropout", "dropout"},
        {"LeakyRelu", "leakyrelu"},
        {"Gemm", "linear"},// not Gemm, but Matmul (ambiguous stuff)
        {"MatMul", "linear"},
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

    for (AttributeProto attribute:node.attribute())
    {
        string attrName = attribute.name();
        std::cout << attribute.name() << "\n";
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
            //if (attribute.s() == "")
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
        //vector<string> attrs = p1[attrName];
        
    }
    map<string, double>::iterator itr;
    for (itr = dimParams.begin(); itr != dimParams.end(); ++itr)
    {
        mappedParams[itr->first] = itr->second;
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
    std::vector<int> dims;
    auto itr = std::begin(weights);
    for (; itr != std::end(weights); itr += 2)
    {
        dims.push_back((*itr).dims(0));
    }
    itr -= 2;
    dims.push_back((*itr).dims(1));
    return dims;
}

void extractWeights2(GraphProto& graph, arma::mat& weightMatrix)
{
    auto& weights = graph.initializer();
    int totalWeights = 0;
    int weightDims[weights.size()];
    int wtNumber = 0;
    for(TensorProto weight:weights)
    {
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
    for(TensorProto weight:weights)
    {
        //cout << "\nWeight type: " << weight.data_type() << " Weight size: " << weight_size << "\n";
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

void extractWeights(GraphProto& graph, arma::mat& weightMatrix)
{
    auto weights = graph.initializer();// change the auto later
    int count = 0;
    int totalWeights = 0;
    std::vector<int> dims = findWeightDims(graph.initializer());
    std::vector<int>::iterator itr = dims.begin();
    while (itr < dims.end() - 1)
    {
        totalWeights += *itr * *(++itr) + *itr;// product of the no. of biases in consecutive layers (hence no. of weights) plus the number of biases
    }
    cout << "Total weights: " << totalWeights << "\n";
    weightMatrix.set_size(totalWeights, 1);
    for(TensorProto& weight:weights)
    {
        int weight_size = weight.dims(0);
        for (int i = 1; i < weight.dims_size(); ++i)
        {
            weight_size *= weight.dims(i);
        }
        cout << "\nWeight type: " << weight.data_type() << " Weight size: " << weight_size << "\n";
        if (weight.data_type() == 1)// for float
        {
            for (int i = 0; i < weight_size; ++i)
            {
                weightMatrix(count++, 0) = weight.float_data()[i];
                //cout << "\nWeight " << count - 1 << " : " << weightMatrix(count - 1, 0);
            }
            continue;
        }
        std::string rawData = weight.raw_data();
        const char* ws = rawData.c_str();
        for (int i = 0; i < weight_size * 4 - 4; i += 4)
        {
            float wt;
            char t[] = {ws[i], ws[i+1], ws[i+2], ws[i+3]};
            memcpy(&wt, &t, sizeof(float));
            weightMatrix(count++, 0) = wt;
        }
    }
}

FFN<> generateModel(GraphProto& graph)
{
    FFN<> mod;
    std::vector<int> dims = findWeightDims(graph.initializer());
    std::vector<int>::iterator itr = dims.begin();
    for (NodeProto node:graph.node())
    {
        map<string, double> dimParams;
        if (node.op_type() == "Add" || node.op_type() == "Identity")
            continue;
        
        if (!(node.op_type() == "Add" || node.op_type() == "Relu" || node.op_type() == "Softmax"))// more to added later or an array to be created
        {
            dimParams["insize"] = *itr;
            itr++;
            dimParams["outsize"] = *itr;
        }
        std::cout << "Node type: " << node.op_type() << "\n";
        mod.Add(getLayer(node, dimParams));
    }
    return mod;
}

// void writeModel(FFN& model)
// {

// }

int main()
{
    ModelProto model;
    std::ifstream in("onnx_linear_model.onnx", std::ios_base::binary);
    model.ParseFromIstream(&in);
    in.close();

    GraphProto graph = model.graph();

    FFN<> ffnModel = generateModel(graph);
    //arma::mat temp;
    extractWeights2(graph, ffnModel.Parameters());
    //temp.print();
    ffnModel.Parameters().print();
    //data::Save("linear_mlpack_model.xml", "linear_model", model);
    std::cout<<model.graph().node().size()<<"\n";
    cout << "Dims of weights: " << graph.initializer().size() << "\n";

}
