#include <mlpack/core.hpp>
#include <boost/test/unit_test.hpp>
#include <mlpack/tests/test_tools.hpp>
#include <mlpack/core/data/split_data.hpp>

// Just copy-pasted some common includes from a previously written program, so not too contextual
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>


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
 */
int generateAP(int a, int d, int n)
{
    // This is one kind of dataset I thought I can easily generate and train upon,
    // even images (random noise basically) can be generated in this format and checked
    // for accuracy as an accuracy of even 10% for the torch model and also 10% for the
    // converted mlpack model would mean that the conversion is correct
}

/*
    Generate an object of class Dataset using generateAP()
 */
int generateDataset(Dataset& dataset)
{

}

/*
    Test the similarity in accuracy between the trained torch and onnx model
 */
int testAccuracy()
{

}

/*
    Build similar torch and mlpack models, use the converter to convert the torch model to mlpack
    (torch has a default onnx converter in built) and finally test the accuracy
 */
int trainModel()
{
    // Either this function will create a number of different models or there will be different
    // functions like trainLinearModel(), trainConvolutionalModel(), etc.
}

// Haven't thought of what all to include here, I was playing with the code here
int main()
{
    FFN<> model;
    data::Load("mlpack_linear_model.xml", "mdl", model);
    // arma::mat prms = model.network[1].parameter;
    // prms.print();
    //arma::mat params = model.Parameters();
    //params.print();
    //model.Model()[1].Parameters().print();
    //cout << params.n_cols << " " << params.n_rows << endl;
    //arma::mat resp = model.Responses();
    //resp.print();
    

}
