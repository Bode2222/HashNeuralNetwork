#include <iostream>
#include <chrono>
#include "NeuralNet.h"
using namespace std;

//Note: There's no way Im going to remember this so I'll say it here. A pipe is a thread that executes the feedforward and backprop
//function during training.
//Also note: there is no bucket limiting during hashing
//Also also 

//TODO: implement multithreading, implement convolutional layers

//Convolutoinal Layers:
//Hash based implementation is out the window since every few neurons have the same weights(the filter)
//I need to know the dimensions of the image in the previous layer for convolution to happen

//Potential problems
//Q: What happens if no output neuron is chosen? The weights can't update (cuz they only update along chosen neurons) 
//so what happens to the network???
//A: Minimize this chance by reducing the number of bits and increasing the number of tables
//Q: What if I set a variable to one thing in one thread and to another thing in another thread?
//A: this should not matter (at least to my 'active' vector in each neuron) as when all the threads are combined, 
//backprop has already been done (and the 'active' vector is no longer needed).
//Q: In the back pass, do you calculate the cost for all output neurons or only the active ones?
//A: I could be wrong but i dont think in a given instance Im not supposed to touch non active nodes, so Only active ones
//Q: Why does the limit of my weight value (as in how high or low it can go) change with the number of tables I have???
//A: It doesnt, it changes with the number of neurons I have

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