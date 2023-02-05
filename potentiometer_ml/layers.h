/*
    nn_linear class의 forward함수의 파라미터들 중 가중치는 1차원 배열로 reshape한 후 받는 다고 가정
    추후 weight 파라미터를 어떤 데이터 형식으로 받는 지 알게 될 경우 변경
*/

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


#endif
