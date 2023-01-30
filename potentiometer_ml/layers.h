#ifndef LAYERS_H
#define LAYERS_H

class nn_relu
{
public:
	void forward(double* input_sum);
private:
};

class nn_sigmoid
{
public:
    void forward(double* input_sum);
};


class nn_softmax
{
public:
    void forward(double* input_sum);
};


#endif LAYERS_H
