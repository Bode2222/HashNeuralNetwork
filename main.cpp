#include <iostream>
#include <chrono>
#include "NeuralNet.h"
using namespace std;

//Note: There's no way Im going to remember this so I'll say it here. A pipe is a thread that executes the feedforward and backprop
//function during training.
//Also note: there is no bucket limiting during hashing
//Also also 

//TODO: implement multithreading

//Potential problems
//Q: What happens if no output neuron is chosen? The weights can't update (cuz they only update along chosen neurons) 
//so what happens to the network???
//A: Minimize this chance by reducing the number of bits and increasing the number of tables

//WHAT DID I DO?! In case the program stops working:->
//It works.... For now.

int main() {
	srand(0);//time(0)

	//Neural Network architecture
	vector<Layer> layout;
	layout.push_back(Layer(DENSE, 02, NONE, 0, 1, 2));
	layout.push_back(Layer(DENSE, 02, TANH, 0, 1, 2));
	layout.push_back(Layer(DENSE, 02, SOFTMAX, 0, 1, 2));
	NeuralNet myNet(layout);

	//Training Data
	vector<vector<float>> inputBatch = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	vector<vector<float>> outputBatch = { {1, 0}, {0, 1}, {0, 1}, {1, 0} };
	int chosenInputIndex = 1;

	if (!myNet.load("OrGate.hnn")) {
		cout << "Training network" << endl;
		//train network
		for (int i = 0; i < 10000; i++) {
			myNet.train(inputBatch, outputBatch);
		}
		//myNet.save("OrGate.hnn");
	}
	else {
		cout << "Loading Network" << endl;
		myNet.save("OrGate.hnn");
	}

	//Convolution Unit Test
	vector<float> filter1 = { 0.5, 0.25, 0.75, 1 };
	vector<float> image1 = { 1, 3, 9, 2, 6, 4, 8, 7, 4.5, 7.5, 6, 9 };

	Image filter;
	filter.val = filter1;
	filter.xDim = 2;
	filter.yDim = 2;

	Image image;
	image.val = image1;
	image.xDim = 3;
	image.yDim = 4;

	vector<float> result = Util::Convolve(image, filter);

	for (int i = 0; i < result.size(); i++) {
		cout << result[i] << " ";
	}
	cout << endl;

	/*Test Net with input*/
	cout << "Error: " << myNet.getError() << endl;
	cout << "Enter 2 numbers, and -1 to quit" << endl;
	float a = 0;
	float b = 0;
	while (a != -1) {
		cin >> a;
		if (a == -1) break;
		cin >> b;
		cout << a << " " << b << ": ";
		vector<float> result = { a, b };
		myNet.feedForward(result);
		myNet.printOutput();
	}
	/*******************************/

	cout << "Program Finished" << endl;
	cin.get();
	return 0;
}