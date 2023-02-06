#include "layers.h"

#ifndef NETWORK_H
#define NETWORK_H

class nn_2layermlp {
public:
    nn_2layermlp();
    nn_2layermlp(const int input_dim, const int hidden_dim = 3, const int output_dim = 3);
    ~nn_2layermlp();

    double* forward(const double* input, double** weight);
    void set_potentiometer(double** w);
private:
    nn_linear linear1;
    nn_linear linear2;
    nn_sigmoid activ;
    nn_softmax classifier;
    nn_binarize binarize;
    double** potentiometer; // 가변저항 가중치
};



#endif
