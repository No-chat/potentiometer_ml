#ifndef LAYERS_H
#define LAYERS_H
#include <cmath>

class nn_sigmoid {
public:
    void forward(double* input_sum);
};


class nn_softmax {
public:
    void forward(double* input_sum);
};




#endif LAYERS_H
