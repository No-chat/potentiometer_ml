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