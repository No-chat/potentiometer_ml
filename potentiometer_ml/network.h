#include "layers.h"

#ifndef NETWORK_H
#define NETWORK_H

class nn_2layermlp {
public:
    nn_2layermlp();
    nn_2layermlp(const int input_dim,const int output_dim);
    double* forward(const double* input,const double* weight);
    
private:
    nn_linear linear1;
    nn_linear linear2;
    nn_sigmoid activ;
    nn_softmax classifier;
    double* potentiometer; //가변저항 weight
};



#endif NETWORK_H
