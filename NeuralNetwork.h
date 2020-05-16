#pragma once
#include <vector>
using namespace std;

struct Layer;
struct Neuron;

struct OneOutput {
	OneOutput();
	OneOutput(float val1, int index1);
	float val;//value of the output
	int index;//which output it is
};

class NeuralNetwork
{
public:
	virtual void trainWithOneOutput(const vector<vector<float>>& inputs, const vector<OneOutput>& outputs) =0;
	//Updates the weights in the network
	virtual void train(const vector<vector<float>>& inputs, const vector<vector<float>>& outputs) =0;
	//Used for forward passes through the network
	virtual void feedForward(const vector<float>& input) = 0;
	//returns a vector of the output. Used after a forward pass (ie feedforward function)
	virtual vector<float> getOutput() = 0;
	//returns the max output index after a forward pass
	virtual int getMaxOutputIndex() = 0;
	//returns the max output after a forward pass
	virtual float getMaxOutput() = 0;
	//Prints output in a straight line
	virtual void printOutput() = 0;

	virtual void save(string) =0;
	virtual bool load(string) =0;
};


