#include "NeuralNetUtil.h"

Image::Image(int x, int y) {
	xDim = x; yDim = y;
	val = vector<float>(x * y);
	for (int i = 0; i < x * y; i++) {
		val[i] = (rand() % 10000) / 5000.f - 1.f;
	}
}

Image::Image(int x, int y, vector<float>& vals) {
	xDim = x; yDim = y;
	val = vals;
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

Layer::Layer(LayerType l, int filterx, int filtery, int numOfFilters, ActivationFunction func, int previmageLength, int previmageWidth, int prevImageDepthorNumOfFilters, bool zPad) {
	layType = l;

	for (int i = 0; i < numOfFilters; i++) {
		filters.push_back(Image(filterx, filtery));
	}
	actFunc = func;
	prevImgLen = previmageLength;
	prevImgWid = previmageWidth;
	prevImgDepth = prevImageDepthorNumOfFilters;
	zeroPad = zPad;

	/*Set size of layer and biases*/
	if (zeroPad) {
		imgLen = prevImgLen;
		imgWid = prevImgLen;
		convoBias = vector<float>(imgLen * imgWid * filters.size());
		if (maxPoolStride != -1) {
			float tempX = 1.f * imgLen / maxPoolStride;
			float tempY = 1.f * imgWid / maxPoolStride;
			int dimX = (tempX > (int)tempX) ? tempX + 1 : tempX;
			int dimY = (tempY > (int)tempY) ? tempY + 1 : tempY;
			mySize = dimX * dimY * filters.size();
		}
		else {
			mySize = imgLen * imgWid * filters.size();
		}
	}
	else {
		imgLen = prevImgLen - filters[0].xDim + 1;
		imgWid = prevImgLen - filters[0].yDim + 1;
		convoBias = vector<float>(imgLen * imgWid * filters.size());
		if (maxPoolStride != -1) {
			float tempX = 1.f * imgLen / maxPoolStride;
			float tempY = 1.f * imgWid / maxPoolStride;
			int dimX = (tempX > (int)tempX) ? tempX + 1 : tempX;
			int dimY = (tempY > (int)tempY) ? tempY + 1 : tempY;
			mySize = dimX * dimY * filters.size();
		}
		else {
			mySize = imgLen * imgWid * filters.size();
		}
	}
	/*********************/
	//Bias
	mySize++;

	Util::Randomize(convoBias);
}

Layer::Layer(LayerType l, int filterx, int filtery, int numOfFilters, ActivationFunction func, int previmageLength, int previmageWidth, int prevImageDepthorNumOfFilters, bool zPad, int maxPoolFilterXYStride) {
	layType = l;

	for (int i = 0; i < numOfFilters; i++) {
		filters.push_back(Image(filterx, filtery));
	}
	actFunc = func;
	prevImgLen = previmageLength;
	prevImgWid = previmageWidth;
	prevImgDepth = prevImageDepthorNumOfFilters;
	zeroPad = zPad;

	maxPoolx = maxPoolFilterXYStride;
	maxPooly = maxPoolx;
	maxPoolStride = maxPoolx;

	/*Set size of layer*/
	if (zeroPad) {
		imgLen = prevImgLen;
		imgWid = prevImgLen;
		convoBias = vector<float>(imgLen * imgWid * filters.size());
		if (maxPoolStride != -1) {
			float tempX = 1.f * imgLen / maxPoolStride;
			float tempY = 1.f * imgWid / maxPoolStride;
			int dimX = (tempX > (int)tempX) ? tempX + 1 : tempX;
			int dimY = (tempY > (int)tempY) ? tempY + 1 : tempY;
			mySize = dimX * dimY * filters.size();
		}
		else {
			mySize = imgLen * imgWid * filters.size();
		}
	}
	else {
		imgLen = prevImgLen - filters[0].xDim + 1;
		imgWid = prevImgLen - filters[0].yDim + 1;
		convoBias = vector<float>(imgLen * imgWid * filters.size());
		if (maxPoolStride != -1) {
			float tempX = 1.f * imgLen / maxPoolStride;
			float tempY = 1.f * imgWid / maxPoolStride;
			int dimX = (tempX > (int)tempX) ? tempX + 1 : tempX;
			int dimY = (tempY > (int)tempY) ? tempY + 1 : tempY;
			mySize = dimX * dimY * filters.size();
		}
		else {
			mySize = imgLen * imgWid * filters.size();
		}
	}
	/*********************/
	//Bias
	mySize++;

	Util::Randomize(convoBias);
}

Layer::Layer(LayerType l, int filterx, int filtery, int numOfFilters, ActivationFunction func, bool zPad, Layer* prevLayer) {
	try {
		if (prevLayer->getLayerType() != CONVO) {
			cout << prevLayer->getLayerType() << endl;
			throw - 3;
		}
	}
	catch (int e) {
		cout << "The layer pointer passed into the constructor of a convolutional layer isnt a convolutional layer. This is a fatal error. check your constructors" << endl;
	}

	layType = l;
	for (int i = 0; i < numOfFilters; i++) {
		filters.push_back(Image(filterx, filtery));
	}

	actFunc = func;
	zeroPad = zPad;
	prevImgDepth = prevLayer->filters.size();

	//Calculate dimensions of prev layer
	//If previous layer was zero padded
	if (prevLayer->zeroPad) {
		//If the prev layer was max pooled
		if (prevLayer->maxPoolStride != -1) {
			int PIL, PIW;
			PIL = prevLayer->prevImgLen;
			PIW = prevLayer->prevImgWid;
			float t = 1.f * PIL / prevLayer->maxPoolStride;
			float u = 1.f * PIW / prevLayer->maxPoolStride;
			prevImgLen = (t > (int)t) ? t + 1 : t;
			prevImgWid = (u > (int)u) ? u + 1 : t;
		}
		//else
		else {
			prevImgLen = prevLayer->prevImgLen;
			prevImgWid = prevLayer->prevImgWid;
		}

	}
	//If it was not zero padded
	else {
		//If the prev layer was max pooled
		if (prevLayer->maxPoolStride != -1) {
			int PIL, PIW;
			PIL = prevLayer->prevImgLen - prevLayer->filters[0].xDim + 1;
			PIW = prevLayer->prevImgWid - prevLayer->filters[0].xDim + 1;
			float t = 1.f * PIL / prevLayer->maxPoolStride;
			float u = 1.f * PIW / prevLayer->maxPoolStride;
			prevImgLen = (t > (int)t) ? t + 1 : t;
			prevImgWid = (u > (int)u) ? u + 1 : t;
		}
		//else
		else {
			prevImgLen = prevLayer->prevImgLen - prevLayer->filters[0].xDim + 1;
			prevImgWid = prevLayer->prevImgWid - prevLayer->filters[0].xDim + 1;
		}
	}

	/*Set size of layer*/
	if (zeroPad) {
		imgLen = prevImgLen;
		imgWid = prevImgLen;
		convoBias = vector<float>(imgLen * imgWid * filters.size());
		if (maxPoolStride != -1) {
			float tempX = 1.f * imgLen / maxPoolStride;
			float tempY = 1.f * imgWid / maxPoolStride;
			int dimX = (tempX > (int)tempX) ? tempX + 1 : tempX;
			int dimY = (tempY > (int)tempY) ? tempY + 1 : tempY;
			mySize = dimX * dimY * filters.size();
		}
		else {
			mySize = imgLen * imgWid * filters.size();
		}
	}
	else {
		imgLen = prevImgLen - filters[0].xDim + 1;
		imgWid = prevImgLen - filters[0].yDim + 1;
		convoBias = vector<float>(imgLen * imgWid * filters.size());
		if (maxPoolStride != -1) {
			float tempX = 1.f * imgLen / maxPoolStride;
			float tempY = 1.f * imgWid / maxPoolStride;
			int dimX = (tempX > (int)tempX) ? tempX + 1 : tempX;
			int dimY = (tempY > (int)tempY) ? tempY + 1 : tempY;
			mySize = dimX * dimY * filters.size();
		}
		else {
			mySize = imgLen * imgWid * filters.size();
		}
	}
	/*********************/
	//Bias
	mySize++;

	Util::Randomize(convoBias);
}

Layer::Layer(LayerType l, int filterx, int filtery, int numOfFilters, ActivationFunction func, bool zPad, int maxPoolxystride, Layer* prevLayer) {
	try {
		if (prevLayer->getLayerType() != CONVO) {
			cout << prevLayer->getLayerType() << endl;
			throw - 3;
		}
	}
	catch (int e) {
		cout << "The layer pointer passed into the constructor of a convolutional layer isnt a convolutional layer. This is a fatal error. check your constructors" << endl;
	}

	layType = l;
	for (int i = 0; i < numOfFilters; i++) {
		filters.push_back(Image(filterx, filtery));
	}

	actFunc = func;
	zeroPad = zPad;
	prevImgDepth = prevLayer->filters.size();

	//Calculate dimensions of prev layer
	//If previous layer was zero padded
	if (prevLayer->zeroPad) {
		//If the prev layer was max pooled
		if (maxPoolStride != -1) {
			int PIL, PIW;
			PIL = prevLayer->prevImgLen;
			PIW = prevLayer->prevImgWid;
			float t = 1.f * PIL / prevLayer->maxPoolStride;
			float u = 1.f * PIW / prevLayer->maxPoolStride;
			prevImgLen = (t > (int)t) ? t + 1 : t;
			prevImgWid = (u > (int)u) ? u + 1 : t;
		}
		//else
		else {
			prevImgLen = prevLayer->prevImgLen;
			prevImgWid = prevLayer->prevImgWid;
		}

	}
	//If it was not zero padded
	else {
		//If the prev layer was max pooled
		if (prevLayer->maxPoolStride != -1) {
			int PIL, PIW;
			PIL = prevLayer->prevImgLen - prevLayer->filters[0].xDim + 1;
			PIW = prevLayer->prevImgWid - prevLayer->filters[0].xDim + 1;
			float t = 1.f * PIL / prevLayer->maxPoolStride;
			float u = 1.f * PIW / prevLayer->maxPoolStride;
			prevImgLen = (t > (int)t) ? t + 1 : t;
			prevImgWid = (u > (int)u) ? u + 1 : t;
		}
		//else
		else {
			prevImgLen = prevLayer->prevImgLen - prevLayer->filters[0].xDim + 1;
			prevImgWid = prevLayer->prevImgWid - prevLayer->filters[0].xDim + 1;
		}
	}

	maxPoolx = maxPoolxystride;
	maxPooly = maxPoolx;
	maxPoolStride = maxPoolx;

	/*Set size of layer*/
	if (zeroPad) {
		imgLen = prevImgLen;
		imgWid = prevImgLen;
		convoBias = vector<float>(imgLen * imgWid * filters.size());
		if (maxPoolStride != -1) {
			float tempX = 1.f * imgLen / maxPoolStride;
			float tempY = 1.f * imgWid / maxPoolStride;
			int dimX = (tempX > (int)tempX) ? tempX + 1 : tempX;
			int dimY = (tempY > (int)tempY) ? tempY + 1 : tempY;
			mySize = dimX * dimY * filters.size();
		}
		else {
			mySize = imgLen * imgWid * filters.size();
		}
	}
	else {
		imgLen = prevImgLen - filters[0].xDim + 1;
		imgWid = prevImgLen - filters[0].yDim + 1;
		convoBias = vector<float>(imgLen * imgWid * filters.size());
		if (maxPoolStride != -1) {
			float tempX = 1.f * imgLen / maxPoolStride;
			float tempY = 1.f * imgWid / maxPoolStride;
			int dimX = (tempX > (int)tempX) ? tempX + 1 : tempX;
			int dimY = (tempY > (int)tempY) ? tempY + 1 : tempY;
			mySize = dimX * dimY * filters.size();
		}
		else {
			mySize = imgLen * imgWid * filters.size();
		}
	}
	/*********************/
	//Bias
	mySize++;

	Util::Randomize(convoBias);
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
				//result[yOffset * (image.xDim - filter.xDim + 1) + xOffset] /= filterSum;
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

vector<float> Util::MaxPool(Image& image, int xDim, int yDim, int stride, vector<int>& maxIndexes) {
	int xOffset = 0, yOffset = 0;
	float x = 1.f * image.xDim / stride;
	float y = 1.f * image.yDim / stride;
	int resultX = (x > (int)x) ? x + 1 : x;
	int resultY = (y > (int)y) ? y + 1 : y;
	vector<float> result(resultX * resultY);
	maxIndexes = vector<int>(resultX * resultY);

	//Pool
	for (yOffset = 0; yOffset < image.yDim; yOffset += stride) {
		for (xOffset = 0; xOffset < image.xDim; xOffset += stride) {
			float max = image.val[(yOffset)*image.xDim + (xOffset)];
			int maxIndex = (yOffset)*image.xDim + (xOffset);
			for (int i = 0; i < yDim; i++) {
				for (int j = 0; j < xDim; j++) {
					if ((yOffset + i) < image.yDim && (xOffset + j) < image.xDim) {
						if (image.val[(yOffset + i) * image.xDim + (xOffset + j)] > max) {
							max = image.val[(yOffset + i) * image.xDim + (xOffset + j)];
							maxIndex = (yOffset + i) * image.xDim + (xOffset + j);
						}
					}
				}
			}
			result[(yOffset / stride) * resultX + (xOffset / stride)] = max;
			maxIndexes[(yOffset / stride) * resultX + (xOffset / stride)] = maxIndex;
		}
	}
	return result;
}

void Util::Randomize(vector<float>& arr) {
	for (int i = 0; i < arr.size(); i++) {
		arr[i] = (rand() % 10000) / 5000 - 1.f;
	}
}

Layer Util::Dense(int layerSize, ActivationFunction func) {
	return Layer(DENSE, layerSize, func);
}

Layer Util::Dense(int layerSize, ActivationFunction func, int neuLim) {
	return Layer(DENSE, layerSize, func, neuLim);
}

Layer Util::Dense(int layerSize, ActivationFunction func, int Bits, int Tables) {
	return Layer(DENSE, layerSize, func, Bits, Tables);
}

Layer Util::Dense(int layerSize, ActivationFunction func, int Bits, int Tables, int neuLim) {
	return Layer(DENSE, layerSize, func, Bits, Tables, neuLim);
}

Layer Util::Convo(int filterx, int filtery, int numOfFilters, ActivationFunction func, bool zeroPad, Layer* prevLayer) {
	return Layer(CONVO, filterx, filtery, numOfFilters, func, zeroPad, prevLayer);
}

Layer Util::Convo(int filterx, int filtery, int numOfFilters, ActivationFunction func, bool zeroPad, int maxPoolxyStride, Layer* prevLayer) {
	return Layer(CONVO, filterx, filtery, numOfFilters, func, zeroPad, maxPoolxyStride, prevLayer);
}

Layer Util::Convo(int filterx, int filtery, int numOfFilters, ActivationFunction func, int previmageLength, int previmageWidth, int prevImageDepthorNumOfFilters, bool zeroPad) {
	return Layer(CONVO, filterx, filtery, numOfFilters, func, previmageLength, previmageWidth, prevImageDepthorNumOfFilters, zeroPad);
}

Layer Util::Convo(int filterx, int filtery, int numOfFilters, ActivationFunction func, int previmageLength, int previmageWidth, int prevImageDepthorNumOfFilters, bool zeroPad, int maxPoolFilterXYStride) {
	return Layer(CONVO, filterx, filtery, numOfFilters, func, previmageLength, previmageWidth, prevImageDepthorNumOfFilters, zeroPad, maxPoolFilterXYStride);
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
	//Calculated as part of last layer backprop calculations
	return 1;
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

void Util::transpose(Image& image) {
	for (int i = 0; i < image.yDim; i++) {
		for (int j = i; j < image.xDim; j++) {
			float temp = image.val[i * image.xDim + j];
			image.val[i * image.xDim + j] = image.val[j * image.xDim + i];
			image.val[j * image.xDim + i] = temp;
		}
	}
}

void Util::reverseColumns(Image& image) {
	for (int i = 0; i < image.xDim; i++) {
		for (int j = 0, k = image.xDim - 1; j < k; j++, k--) {
			float temp = image.val[k * image.xDim + i];
			image.val[k * image.xDim + i] = image.val[j * image.xDim + i];
			image.val[j * image.xDim + i] = temp;
		}
	}
}

void Util::rotate180(Image& image) {
	transpose(image);
	reverseColumns(image);
	transpose(image);
	reverseColumns(image);
}
