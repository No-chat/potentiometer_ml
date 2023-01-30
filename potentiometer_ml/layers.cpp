#include "layers.h"
#include <iostream>
#include <cmath>

using namespace std;

void nn_relu::forward(double* input_sum)
{
	int col = sizeof(input_sum) / sizeof(double);

	for (int col_index = 0; col_index < col; col_index)
	{
		if (input_sum[col_index] < 0.0)
		{
			input_sum[col_index] = 0;
		}
	}
}

void nn_sigmoid::forward(double* input_sum)
{
    int col=sizeof(*input_sum)/sizeof(double);
    
    for(int col_index=0;col_index<col;col_index++)
    {
       input_sum[col_index]=1/(1+exp(-input_sum[col_index]));
    }
}

void nn_softmax::forward(double *input_sum)
{
    int col=sizeof(*input_sum)/sizeof(double);
    
    double sum=0;
    
    for(int col_index=0;col_index<col;col_index++)
    {
        sum+=exp(input_sum[col_index]);
    }
    
    for(int col_index=0;col_index<col;col_index++)
    {
        input_sum[col_index]=exp(input_sum[col_index])/sum;
    }
}

