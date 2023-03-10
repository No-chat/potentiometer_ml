#include "network.h"
#include <iostream>

nn_2layermlp::nn_2layermlp()
{
    linear1=nn_linear();
    linear2=nn_linear();
    activ=nn_sigmoid();
    classifier=nn_softmax();
    binarize=nn_binarize();
    potentiometer=0;
}

nn_2layermlp::nn_2layermlp(const int input_dim,const int hidden_dim, const int output_dim)
{
    linear1=nn_linear(input_dim,hidden_dim);
    linear2=nn_linear(hidden_dim,output_dim);
    activ=nn_sigmoid();
    classifier=nn_softmax();
    binarize=nn_binarize();
    potentiometer=0;
}

nn_2layermlp::~nn_2layermlp()
{
    if (potentiometer != 0) delete[] potentiometer;
}

double* nn_2layermlp::forward(const double *input,double** weight)
{
    int output_node1=linear1.get_output_nodes();
    int output_node2=linear2.get_output_nodes();

    double* sum1;
    double* sum2;
    sum1=linear1.forward(input,weight);
    activ.forward(sum1, output_node1);
    binarize.forward(sum1, output_node1);
    
    for (int index=0; index<output_node1; index++)
    {
        std::cout<<sum1[index]<<' ';
    }
    
    std::cout<<std::endl;
    
    sum2=linear2.forward(sum1,potentiometer);
    classifier.forward(sum2,output_node2);
    
    
    
    delete[] sum1;

    return sum2;
}

void nn_2layermlp::set_potentiometer(double** w)
{
    potentiometer = w;
}
