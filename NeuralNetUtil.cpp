#include "NeuralNetUtil.h"

Image::Image(int x, int y) {
	xDim = x; yDim = y;
	val = vector<float>(x * y);
	for (int i = 0; i < x * y; i++) {
		val[i] = (rand() % 10000) / 5000 - 1.f;
	}
}

Neuron::Neuron(unsigned batchSize) {
	SetVars(batchSize);
}

Neuron::Neuron(unsigned batchSize, vector<float> w) {
	SetVars(batchSize);
	weight = w;
}

Neuron::Neuron(vector<float> w) {
	weight = w;
}

float Neuron::maxW = 8;

int Neuron::floatToInt(float f) {
	//Whatever the bigNum is when its multiplied by maxweight it should be under max rand(around 32767)
	float bigNum = 1e3;

	if (f >= maxW) {
		maxW = f;
	}
	else if (f <= -maxW) {
		maxW = -f;
	}

	f *= bigNum;
	unsigned a = f + maxW * bigNum;
	return a;
}

void Neuron::SetVars(int batchSize) {
	active = vector<unsigned>(batchSize / (sizeof(int) * 8) + 1);
	gradient = vector<float>(batchSize);
	activation = vector<float>(batchSize);
}

void Neuron::pushActive(bool num) {
	if (actCount == 0) { active.push_back(num); }
	else { active.back() = pow(2, actCount) * (int)num + active.back(); }
	actCount = (actCount + 1) % (sizeof(int) * 8);
}

bool Neuron::getActive(unsigned loc) {
	if (loc >= active.size() * sizeof(int) * 8) { cout << "bit out of bounds, getactive(x)." << endl; return false; }
	unsigned a = loc % (sizeof(int) * 8);
	loc = loc / (sizeof(int) * 8);
	a = pow(2, a);
	return (active[loc] & a);
}

void Neuron::setActive(unsigned loc, bool num) {
	if (loc >= active.size() * sizeof(int) * 8) { cout << "bit out of bounds, getactive(x)." << endl; return; }
	unsigned a = loc % (sizeof(int) * 8);
	loc = loc / (sizeof(int) * 8);
	a = pow(2, a);

	if ((active[loc] & a) == num) return;
	else {
		if (num) active[loc] += a;
		else active[loc] -= a;
	}
}

Layer::Layer(LayerType l, int layerSize, ActivationFunction func) {
	layType = l;
	actFunc = func;
	mySize = layerSize + 1;//plus bias
	HashTable = SimHash(bits, tables);
}

Layer::Layer(LayerType l, int layerSize, ActivationFunction func, int neuLim) {
	layType = l;
	actFunc = func;
	mySize = layerSize + 1;
	HashTable = SimHash(bits, tables);
	neuronLimit = neuLim;
}

Layer::Layer(LayerType l, int layerSize, ActivationFunction func, int Bits, int Tables) {
	layType = l;
	actFunc = func;
	mySize = layerSize + 1;
	bits = Bits;
	tables = Tables;
	HashTable = SimHash(bits, tables);
}

Layer::Layer(LayerType l, int layerSize, ActivationFunction func, int Bits, int Tables, int neuLim) {
	layType = l;
	actFunc = func;
	mySize = layerSize + 1;
	bits = Bits;
	tables = Tables;
	HashTable = SimHash(bits, tables);
	neuronLimit = neuLim;
}

vector<float> Layer::inputAt(int x) {
	vector<float> result;
	for (int i = 0; i < mySize; i++) {
		if (neuron[i].getActive(x)) result.push_back(neuron[i].activation[x]);
		else result.push_back(0);
	}
	return result;
}

vector<unsigned> Layer::intInputAt(int x) {
	vector<unsigned> result;
	//Dont add the bias
	for (int i = 0; i < mySize - 1; i++) {
		if (neuron[i].getActive(x)) {
			result.push_back(Neuron::floatToInt(neuron[i].activation[x]));
		}
		else { result.push_back(0); }
	}
	return result;
}

float Layer::activate(float x) {
	switch (actFunc) {
	case TANH:
		return Layer::TanhActivate(x);
		break;
	case RELU:
		return Layer::ReluActivate(x);
		break;
	case SIGMOID:
		return Layer::SigmoidActivate(x);
		break;
	case SOFTMAX:
		return Layer::SoftmaxActivate(x);
		break;
	case NONE:
		return Layer::NoneActivate(x);
		break;
	}
}

float Layer::dActivate(float x) {
	switch (actFunc) {
	case TANH:
		return Layer::TanhDActivate(x);
		break;
	case RELU:
		return Layer::ReluDActivate(x);
		break;
	case SIGMOID:
		return Layer::SigmoidDActivate(x);
		break;
	case SOFTMAX:
		return Layer::SoftmaxDActivate(x);
		break;
	case NONE:
		return Layer::NoneDActivate(x);
		break;
	}
}

float Layer::SigmoidActivate(float x) {
	float expon = exp(x);
	float ans = expon / (expon + 1);
	return ans;
}

float Layer::SigmoidDActivate(float x) {
	return x * (1 - x);
}

float Layer::SoftmaxActivate(float x) {
	float expon = exp(x);
	return expon;
}

float Layer::SoftmaxDActivate(float x) {
	return x * (1 - x);
}

float Layer::ReluActivate(float x) {
	if (x > 0) return x;
	return 0;
}

float Layer::ReluDActivate(float x) {
	if (x > 0) return 1;
	return 0;
}

float Layer::TanhActivate(float x) {
	return tanh(x);
}

float Layer::TanhDActivate(float x) {
	return 1 - x * x;
}

vector<float> Util::Convolve(Image& image, Image& filter) {
	vector<float> result((image.xDim - filter.xDim + 1) * (image.yDim - filter.yDim + 1));
	//If filter x or y is outside the image, throw error/exception/stop the program
	try {
		if (filter.xDim > image.xDim || filter.yDim > image.yDim) {
			throw - 2;
		}
		//else
		int yOffset = 0;
		int xOffset = 0;
		float filterSum = 0;
		//Calculate the sum of values in the filter, used to normalize the values of the dot product btw the filter and the image
		for (int i = 0; i < filter.xDim * filter.yDim; i++) {
			filterSum += filter.val[i];
		}

		//Convolve
		while (yOffset + filter.yDim <= image.yDim) {
			xOffset = 0;
			while (xOffset + filter.xDim <= image.xDim) {
				for (int i = 0; i < filter.yDim; i++) {
					for (int j = 0; j < filter.xDim; j++) {
						result[yOffset * (image.xDim - filter.xDim + 1) + xOffset] += filter.val[i * filter.xDim + j] * image.val[(i + yOffset) * image.xDim + (j + xOffset)];
					}
				}
				result[yOffset * (image.xDim - filter.xDim + 1) + xOffset] /= filterSum;
				xOffset++;
			}
			yOffset++;
		}
	}
	catch (int e) {
		cout << "Error: One or more filter dimensions is larger than their image counterpart. Check convolve function. error num: " << e << endl;
	}

	return result;
}
