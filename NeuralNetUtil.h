#pragma once
#include <vector>
#include <iostream>
#include "SimHash.h"
using namespace std;

struct Neuron {
	Neuron() {};
	Neuron(unsigned batchSize);
	Neuron(unsigned batchsize, vector<float> weight);
	Neuron(vector<float> weight);

	vector<float> weight;
	vector<float> gradient;
	vector<float> activation;

	//Add true or false to the end of the active vector
	void pushActive(bool);
	//returns true or false from a position in the active vector
	bool getActive(unsigned);
	//set a position in the active vector to either true or false
	void setActive(unsigned, bool);
	//converts floats to integers
	static int floatToInt(float);
	//Assigns related values based on batch size
	void SetVars(int bSize);
	static float maxW;
private:
	int actCount = 0;
	//stores neurons activeness in bits
	vector<unsigned> active;
	//static fti FloatConverter;
};

enum ActivationFunction {
	TANH, RELU, SIGMOID, SOFTMAX, NONE
};

enum LayerType {
	DENSE, CONVO, GRU
};

struct Image {
	Image() {};
	Image(int x, int y);
	int xDim, yDim;
	vector<float> val;
};

struct Layer {
	Layer() {};
	//Pass input through activation function
	float activate(float x);
	float dActivate(float x);
	int size() { return mySize; }
	void setSize(int x) { mySize = x; }
	LayerType getLayerType() { return layType; }
	ActivationFunction getActivationFunction() { return actFunc; }
	//Default Hash Table vars are 6 bits, 3 tables and 5 neurons chosen to be multiplied with their weights every time the layer is multiplied through
	Layer(LayerType l, int layerSize, ActivationFunction func);
	Layer(LayerType l, int layerSize, ActivationFunction func, int neuLim);
	Layer(LayerType l, int layerSize, ActivationFunction func, int Bits, int Tables);
	Layer(LayerType l, int layerSize, ActivationFunction func, int Bits, int Tables, int neuLim);
	//Used in creating the first convolutional layer wihtout max pooling. Filter x and y are the dimensions of the filters that convolve over the image\
	The activation function is self explanatory(eg RELU, TANH), the prev image lenght is the x dimension of the input image, the width is the y dimension\
	If the previous image did not have zero padding and was from a convolutional layer, the dimensions are imagex - filterx + 1, repeat for y dimension\
	The previous image depth means the number of color channels if we are taking an rgb/black&white image from the input layer or the number of filters if we are taking\
	the image from a convolutional layer. Zero padding means maintain the image dimensions during convolution by padding the sides of the image with zero.
	Layer(LayerType l, int filterx, int filtery, int numOfFilters, ActivationFunction func, int previmageLength, int previmageWidth, int prevImageDepthorNumOfFilters, bool zeroPad);
	//Used in creating the first convolutional layer with max pooling. max pooling reduces data size. I usually go with a 2 x 2 filter with a 2 stride\
	To know what the other variables do check the other convolutional layer initiator-
	Layer(LayerType l, int filterx, int filtery, int numOfFilters, ActivationFunction func, int previmageLength, int previmageWidth, int prevImageDepthorNumOfFilters, bool zeroPad, int maxPoolFilterX, int maxPoolFilterY, int maxPoolstride);
	//DO NOT USE UNLESS THIS LAYER COMES AFTER A CONVOLUTIONAL LAYER. Used in creating subsequent convolutional layers without max pooling. All the work of calculating image size will be done in code
	Layer(LayerType l, int filterx, int filtery, ActivationFunction func, bool zeroPad);
	///DO NOT USE UNLESS THIS LAYER COMES AFTER A CONVOLUTIONAL LAYER. Used in creating subsequent convolutional layers with max pooling. All the work of calculating image size will be done in code
	Layer(LayerType l, int filterx, int filtery, ActivationFunction func, bool zeroPad, int maxPoolx, int maxPooly, int maxPoolstride);

	//public vars
	int neuronLimit = 99999;
	int bits = 6, tables = 3;
	SimHash HashTable;
	vector<Neuron> neuron;
	vector<float> inputAt(int pipe);
	vector<unsigned> intInputAt(int pipe);

private:
	static float TanhActivate(float x);
	static float SigmoidActivate(float x);
	static float ReluActivate(float x);
	static float SoftmaxActivate(float x);
	static float NoneActivate(float x) { return x; };
	static float TanhDActivate(float x);
	static float SigmoidDActivate(float x);
	static float ReluDActivate(float x);
	static float SoftmaxDActivate(float x);
	static float NoneDActivate(float x) { return 1; }

	//Layer vars
	ActivationFunction actFunc;
	LayerType layType;
	int mySize;
};

class Util {
public:
	static vector<float> Convolve(Image& image, Image& filter);
};
