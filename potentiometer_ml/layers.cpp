#include "layers.h"
#include <math.h>

using namespace std;

nn_linear::nn_linear()
    :input_nodes(0), output_nodes(0)
{

}

nn_linear::nn_linear(const int input_dim, const int output_dim)
{
    input_nodes = input_dim;
    output_nodes = output_dim;
}

double* nn_linear::forward(const double* input, double** weight)
{
    double* output_arr = new double[output_nodes];
    for (int output_index = 0; output_index < output_nodes; output_index++)
    {
        double sum = 0;
        for (int input_index = 0; input_index < input_nodes; input_index++)
        {
            sum += (input[input_index] * weight[input_index][output_index]);
        }
        output_arr[output_index] = sum;
    }

    return output_arr;
}

int nn_linear::get_output_nodes() const
{
    return output_nodes;
}

void nn_relu::forward(double* input_sum, int col)
{
    for (int col_index = 0; col_index < col; col_index++)
	{
		if (input_sum[col_index] < 0.0)
		{
			input_sum[col_index] = 0.0;
		}
	}
}

void nn_sigmoid::forward(double* input_sum, int col)
{
    for (int col_index = 0; col_index < col; col_index++)
    {
        input_sum[col_index] = 1 / (1 + exp(-input_sum[col_index]));
    }
}

void nn_softmax::forward(double* input_sum, int col)
{
    double sum = 0;

    for (int col_index = 0; col_index < col; col_index++)
    {
        sum += exp(input_sum[col_index]);
    }

    for (int col_index = 0; col_index < col; col_index++)
    {
        input_sum[col_index] = exp(input_sum[col_index]) / sum;
    }
}
