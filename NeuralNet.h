#pragma once
#include "NeuralNetUtil.h"
#include "NeuralNetwork.h"
#include <fstream>

//v0.1: Simple feed forward and backprop. Working with xor problem and carTrack problem.
//v0.1.1: Convolutions implemented. Working with digit recognition.
//v0.1.2: Adam optimizer and trainTillError() function implemented.
//v0.2: gru layer implemented and tested. Working with sentiment analysis problem.
//v0.3: Multithread and batchNorm implemented and tested.
//current version: v0.1.2

//ToDo:
//Put stops if the net was initialized without any layers, or if i try to backpropagate without feed forwarding first.
//Make "NextLayerDense()" function work in "DenseBackwardPass()" function like it does in "ConvBackwardPass()"
//Make sure I'm not currently calling the "NextLayerConvo()" function for every neuron in a dense layer, because that would be foolish and there's\
No way I would have made that mistake right?
//I should be making all these edits to the file in the main HashNN program I wrote not here. smh 

//If I stopped programming in an uncompilable state, Where did I stop last?
//Program Complete

class NeuralNet : public NeuralNetwork
{
	//Variables
	int baseT = 50;
	float lambda = 1.1;
	float growthRate = 0.1;
	float adaptiveLearningRateHyperparameter = 0.9;
	float momentumHyperparameter = 0.9;
	//number of hash table updates
	int t = 0;
	//Number of iterations, used to determine the frequency of hashtable updates
	int iter = 0;
	int nextUpdate = baseT;
	float loss;

	bool DEBUG = false;

	vector<Layer> net;
	//Default cost function
	float Cost(float myOutput, float target);
	//Cost function derivative
	float CostDerivative(float myOutput, float target);
	//Print output of the last layer
	void printOutput(int pipe);
	//returns output of the last layer in a specific pipe
	vector<float> getOutput(int pipe);
	//Forward pass through the network
	void feedForward(vector<float> input, int pipe);
	//Back pass through the network
	void BackPropagate(const vector<float>& output, int pipe);
	//backward pass through a dense layer using stochastic gradient descent
	void DenseBackwardPass(int layerIndex, int pipeIndex);
	//Foward pass through a dense layer
	void DenseForwardPass(int layerIndex, int pipeIndex);
	//backward pass through a convolutional layer using stochastic gradient descent
	void ConvBackwardPass(int layerIndex, int pipeIndex);
	//Foward pass through a Convolutional layer
	void ConvForwardPass(int layerIndex, int pipeIndex);
	//Determines which layers get what kind of Back pass. eg dense layers get dense back pass
	void startNetwork(vector<Layer>& layout);
	//Clears the gradients of every weight 
	void clearWeightGradients();
	//Adds together the weight gradients in every pipe then applies them to the weight at the end of a batch
	void applyWeightGradients();

	//Element wise multiplication of two vectors
	float multVec(const vector<float>&, const vector<float>&);
	//Updates all the hashtables
	void UpdateHashTables();
	//Updates number of iterations and updates hash tables
	void HashUpdateTracker();
	//Calculates the gradients of this layer if the next layer is Fully connected(Dense)
	vector<float> GetGradientIfNextLayerDense(int layerIndex, int pipe);
	//Calculates the gradients of this layer if the next layer is convolutional
	vector<float> GetGradientIfNextLayerConvo(int layerIndex, int pipe);

	//Debug functions
	void DebugWeights();
	void DebugWeights(int layer);

	//Load functions
	void LoadPrevNetVersion(ifstream);
	void LoadCurrNetVersion(ifstream);

public:
	void setDebugFlag(bool dbug) { DEBUG = dbug; };
	float getError();
	NeuralNet() {};
	void save(string);
	bool load(string);
	int getLayerSize(int layerIndex);
	void printOutput();
	float getMaxOutput();
	int getMaxOutputIndex();
	vector<float> getOutput();
	NeuralNet(vector<Layer>& layout);
	int size() { return net.size(); }
	void operator=(const NeuralNet& obj);
	int getConvLayerFilterSize(int layerIndex);
	vector<float> getLayerOutput(int layerIndex);
	void feedForward(const vector<float>& input);
	vector<int> getConvLayerImgSize(int layerIndex);
	//Accepts one batch of input/output pair
	void train(const vector<vector<float>>& input, const vector<vector<float>>& output);
	//Accepts entire dataset
	void trainTillError(const vector<vector<float>>& input, const vector<vector<float>>& output, int numOfBatches, int numOfEpochs, float targetError);
	void trainWithOneOutput(const vector<vector<float>>& inputs, const vector<OneOutput>& outputs);
};