#ifndef LAYERS_H
#define LAYERS_H

class nn_linear
{
public:
    nn_linear();
    nn_linear(const int input_dim, const int output_dim);

    int get_output_nodes() const;

    double* forward(const double* input, double** weight);
private:
    int input_nodes;
    int output_nodes;
};

class nn_relu
{
public:
	void forward(double* input_sum, int col);
private:
};

class nn_sigmoid
{
public:
    void forward(double* input_sum, int col);
};


class nn_softmax
{
public:
    void forward(double* input_sum, int col);
};

class nn_binarize
{
public:
    void forward(double* input_sum,int col);
    
};

#endif
