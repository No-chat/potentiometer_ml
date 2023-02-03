#include "network.h"
#include <iostream>
#include <random>

using namespace std;

int main()
{
	const int input_size = 784;
	const int hidden_size = 3;
	const int output_size = 3;

	// 실수 난수 생성
	random_device rd;
	mt19937 gen(rd());
	normal_distribution<double> dis(0, 1);

	// 임의의 784개 input
	double input[input_size];
	for (int col = 0; col < input_size; col++)
	{
		input[col] = dis(gen);
	}

	//from data.cpp
	extern double weight[][hidden_size];
	extern double weight_p[][output_size];

	double** w1 = new double* [input_size];
	double** w2 = new double* [hidden_size];

	for (int row = 0; row < input_size; row++)
	{
		w1[row] = weight[row];
	}

	for (int row = 0; row < hidden_size; row++)
	{
		w2[row] = weight_p[row];
	}


	nn_2layermlp network(input_size, hidden_size, output_size);
	network.set_potentiometer(w2);
	// network객체의 멤버 변수가 w2가 가리키는 공간을 가리키고 w2 pointer변수는 dangling포인터로 만들어 준다
	w2 = 0;

	double* result;

	result = network.forward(input, w1);

	for (int i = 0; i < output_size; i++)
	{
		cout << result[i] << " ";
	}

	delete[] w1;
	
}