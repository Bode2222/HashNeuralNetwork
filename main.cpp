#include <chrono>
#include "NeuralNet.h"
using namespace std;

//Note: 
//There's no way Im going to remember this so I'll say it here. A pipe is a thread that executes the feedforward and backprop functions during training.
//There is no bucket limiting during hashing. Its possible all neurons pack into one bucket and reduce performance.

//TODO: Multithreading
//Multithread the feed forward and backpropagate functions
//FFBP will take a vector input, vector output and pipe number, FFBPWOO will take vector input, vector output, pipe number and output value
//Multithread the outputs of different filters in convolutional layers then add them together
//Edited dense backprop function to use nextlayerdense and nextlayerconvo functions
//Switched to minibatch GD(gradient descent) for dense and convolutional layers
//fix problem caused by non divisible batch sizes

//WHAT DID I DO?! In case the program stops working:->
//in the feed forward function in NeuralNet.cpp, I am trying to join my threads. if i join them in the same loop that theyre created in, it works fine, if I
//Try to join them in a seperate loop however...

int main() {
	//Neural Network architecture
	vector<Layer> layout;
	layout.push_back(Layer(DENSE, 02, NONE, 0, 1, 2));
	layout.push_back(Layer(DENSE, 02, TANH, 0, 1, 2));
	layout.push_back(Layer(DENSE, 01, NONE, 0, 1, 2));
	NeuralNet myNet(layout);

	//Training Data
	vector<vector<float>> inputBatch = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	vector<vector<float>> outputBatch = { {1}, {0}, {0}, {1} };
	int chosenInputIndex = 1;

	if (!myNet.load("OrGate.hnn")) {
		cout << "Training network" << endl;
		//train network
		chrono::system_clock::time_point startTime = chrono::system_clock::now();
		myNet.trainTillError(inputBatch, outputBatch, 1, 3000, 0.0001);
		chrono::system_clock::time_point endTime = chrono::system_clock::now();
		std::chrono::duration<double, std::milli> timeTaken = endTime - startTime;
		cout << "Time taken: " << timeTaken.count() << endl;
		//myNet.save("OrGate.hnn");
	}
	else {
		cout << "Loading Network" << endl;
		myNet.save("OrGate.hnn");
	}

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