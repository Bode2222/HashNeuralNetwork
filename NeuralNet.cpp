#include "NeuralNet.h"

template <class T>
void PrintVec(vector<T> vec) {
	cout << "{ ";
	for (int i = 0; i < vec.size(); i++) {
		cout << vec[i];
		if (i + 1 < vec.size()) cout << ", ";
	}
	cout << " }" << endl;
}

float NeuralNet::Cost(float myOutput, float target) {
	//If im using softmax to classify return cross entropy loss
	if (net[net.size() - 1].getActivationFunction() == SOFTMAX) {
		float temp = -target * log(myOutput);
		return temp;
	}
	//else return mean squared error
	return (myOutput - target) * (myOutput - target);
}

float NeuralNet::CostDerivative(float myOutput, float target) {
	//If im using softmax to classify return (cross entropy loss + softmax) derivative. I derived them both at once so I wont have to use dActivate. 
	//check:http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html for an in depth explanation of how to derive it
	if (net[net.size() - 1].getActivationFunction() == SOFTMAX) {
		//if (target == 1) cout << myOutput << " - " << target << " = " << myOutput - target << endl;
		//else cout << myOutput << " = " << myOutput << endl;
		return (target == 1) ? (myOutput - target) : myOutput;
	}
	//else return mean squared error derivative
	float result = (myOutput - target) * net[net.size() - 1].dActivate(myOutput);
	return result;
}

float NeuralNet::getError() {
	return loss;
}

void NeuralNet::startNetwork(vector<Layer>& layout) {
	net = layout;
	//Remove the bias from the last layer
	net[net.size() - 1].setSize(net[net.size() - 1].size() - 1);

	//for every layer
	for (int i = 0; i < net.size(); i++) {
		vector<vector<unsigned>> weightArray;
		//for every neuron.
		for (int j = 0; j < net[i].size(); j++) {
			//Not(If its the input layer, or its the last neuron and its not the last layer, or if its a convolutional layer)
			if (!(i == 0) && !(j == net[i].size() - 1 && i < net.size() - 1) && !(net[i].getLayerType() == CONVO)) {
				vector<float> weight;
				vector<unsigned> intWeights;
				for (int k = 0; k < net[i - 1].size(); k++) {
					weight.push_back(1.f * (rand() % 10000) / 10000 - 0.5);
					//Don't add the bias weight to the hash table
					if (k + 1 != net[i - 1].size())
						intWeights.push_back(Neuron::floatToInt(weight.back()));
				}
				net[i].neuron.push_back(Neuron(weight));
				weightArray.push_back(intWeights);
			}
			//no need for weights on the input layer or the bias node or the convolutional layer and batchsize is set when training
			else {
				net[i].neuron.push_back(Neuron());
			}
		}
		//dont hash input layer ir convolutional layers
		if (i != 0 && net[i].getLayerType() != CONVO) net[i].HashTable.Hash(weightArray);
	}
}

/*When instantiating the network, instantiate the layers
*then reach into them and individually assign each neuron random weights
*After which hash all neurons of each layer into all tables of each layer
*based on their weights*/
NeuralNet::NeuralNet(vector<Layer>& layout) {
	startNetwork(layout);
}

void NeuralNet::feedForward(vector<float> input, int pipe) {
	//if input doesnt match the input layer, throw error
	if (input.size() != net[0].neuron.size() - 1) {
		cout << "Net input layer size is " << net[0].neuron.size() - 1 << endl;
		cout << "Data size is " << input.size() << endl;
		cout << "Input doesnt match input layer size" << endl;
		return;
	}
	//Put input values into the input layer
	input.push_back(1); //bias
	for (int i = 0; i < input.size(); i++) {
		net[0].neuron[i].activation[pipe] = input[i];
		net[0].neuron[i].setActive(pipe, 1);
	}

	//For every layer apart fromt the input
	for (int i = 1; i < net.size(); i++) {
		if (net[i].getLayerType() == DENSE) DenseForwardPass(i, pipe);
		else if (net[i].getLayerType() == CONVO) ConvForwardPass(i, pipe);
	}
}

void NeuralNet::feedForward(const vector<float>& input) {
	//Set batch size
	if (net[0].neuron[0].activation.size() == 0) {
		for (int i = 0; i < net.size(); i++) {
			net[i].maxNeuronIndex = vector<vector<vector<int>>>(1, vector<vector<int>>(net[i].filters.size(), vector<int>()));
			for (int j = 0; j < net[i].size(); j++) {
				net[i].neuron[j].SetVars(1);
			}
		}
	}
	feedForward(input, 0);
}

void NeuralNet::BackPropagate(const vector<float>& output, int pipe) {
	//If output layer size doesnt match target output size, throw error
	if (output.size() != net[net.size() - 1].neuron.size()) {
		cout << "Output doesnt match output layer size" << endl;
		return;
	}

	int outLayIndex = net.size() - 1;
	//Put output gradient into output layer. Only calculate cost of active neurons
	loss = 0;
	for (unsigned i = 0; i < output.size(); i++) {
		if (net[outLayIndex].neuron[i].getActive(pipe)) {
			net[outLayIndex].neuron[i].gradient[pipe] = CostDerivative(net[outLayIndex].neuron[i].activation[pipe], output[i]);
			loss += Cost(net[outLayIndex].neuron[i].activation[pipe], output[i]);
		}
	}

	//For every layer except the input. update weights and neurons
	if (loss > 0) {
		for (int i = outLayIndex; i > 0; i--) {
			if (net[i].getLayerType() == DENSE) DenseBackwardPass(i, pipe);
			if (net[i].getLayerType() == CONVO) ConvBackwardPass(i, pipe);
		}
	}

}

void NeuralNet::printOutput(int pipe) {
	std::cout << "Output: { ";
	for (int i = 0; i < net[net.size() - 1].size(); i++) {
		cout << net[net.size() - 1].neuron[i].activation[pipe] * net[net.size() - 1].neuron[i].getActive(pipe);
		if (i + 1 != net[net.size() - 1].size()) std::cout << ", ";
	}
	std::cout << " }" << endl;
}

vector<float> NeuralNet::getLayerOutput(int layerIndex) {
	return net[layerIndex].inputAt(0);
}

void NeuralNet::printOutput() {
	printOutput(0);
}

float NeuralNet::multVec(const vector<float>& a, const vector<float>& b) {
	if (a.size() != b.size()) {
		cout << "Vectors to be multiplied didnt have the same size." << endl;
		return -1;
	}
	float sum = 0;
	for (int i = 0; i < a.size(); i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

void NeuralNet::DenseForwardPass(int layerIndex, int pipe) {
	//get neuron indexes by querying this layers' hash tables
	set<int> neurInd = net[layerIndex].HashTable.randQueryTill(net[layerIndex - 1].intInputAt(pipe), net[layerIndex].neuronLimit);

	//Softmax vars
	bool isSoftmax = false;
	float sum = 0;
	if (net[layerIndex].getActivationFunction() == SOFTMAX)
		isSoftmax = true;

	//Calculate the values of the chosen neurons
	for (auto it = neurInd.begin(); it != neurInd.end(); it++) {
		//multiply previous layer values by previous layer weights and pass the val through an activation function. also set active to true
		net[layerIndex].neuron[*it].activation[pipe] = net[layerIndex].activate(multVec(net[layerIndex - 1].inputAt(pipe), net[layerIndex].neuron[*it].weight));
		net[layerIndex].neuron[*it].setActive(pipe, 1);

		if (isSoftmax) {
			sum += net[layerIndex].neuron[*it].activation[pipe];
		}
	}

	//If softmax is the activation function, each activated neuron has to be diveded by the sum
	if (isSoftmax) {
		for (auto it = neurInd.begin(); it != neurInd.end(); it++) {
			net[layerIndex].neuron[*it].activation[pipe] /= sum;
		}
	}

	//bias.
	if (layerIndex + 1 != net.size()) {
		net[layerIndex].neuron[net[layerIndex].size() - 1].activation[pipe] = 1;
		net[layerIndex].neuron[net[layerIndex].size() - 1].setActive(pipe, 1);
	}
}

void NeuralNet::DenseBackwardPass(int layerIndex, int pipe) {
	//Update Gradient of neurons in current layer
	if (layerIndex != net.size() - 1) {
		vector<float> gradientAtThisPipe;
		if (net[layerIndex + 1].getLayerType() != CONVO) {
			gradientAtThisPipe = GetGradientIfNextLayerDense(layerIndex, pipe);
		}
		else {
			gradientAtThisPipe = GetGradientIfNextLayerConvo(layerIndex, pipe);
		}

		if (gradientAtThisPipe.size() != net[layerIndex].neuron.size()) {
			cout << "Gradient dimensions not equal to layer dimensions. check convsgdbackpass funciton. Layer " << layerIndex << endl;
			cout << "Gradient At this pipe: " << gradientAtThisPipe.size() << endl;
			cout << "Actual layer size: " << net[layerIndex].neuron.size() << endl;
			return;
		}
		for (int i = 0; i < gradientAtThisPipe.size(); i++) {
			net[layerIndex].neuron[i].gradient[pipe] = gradientAtThisPipe[i];
		}
	}

	//Update the weights leading into that neuron
	for (int i = 0; i < net[layerIndex].size(); i++) {
		//If the neuron is active and (the neuron is in the last layer or the neuron is not the last neuron in the layer(ie the bias))
		if (net[layerIndex].neuron[i].getActive(pipe) && !(layerIndex != net.size() - 1 && i == net[layerIndex].size() - 1)) {
			for (int j = 0; j < net[layerIndex - 1].size(); j++) {
				//if the weight is coming from an active neuron
				if (net[layerIndex - 1].neuron[j].getActive(pipe)) {
					Neuron* prevNeuron = &net[layerIndex - 1].neuron[j];
					Neuron* curNeuron = &net[layerIndex].neuron[i];
					//weight update is divided by batch size and diminished by growthRate
					net[layerIndex].neuron[i].weightGradient[j][pipe] += prevNeuron->activation[pipe] * net[layerIndex].dActivate(curNeuron->activation[pipe]) * curNeuron->gradient[pipe];
				}
			}
		}
	}
}

void NeuralNet::ConvForwardPass(int layerIndex, int pipe) {
	//Check if my calculated last layer image size is actually the last layer image size
	if (net[layerIndex].prevImgLen * net[layerIndex].prevImgDepth * net[layerIndex].prevImgWid + 1 != net[layerIndex - 1].size()) {
		cout << "Calculated prev layer is not the size of actual prev layer, check conv forward pass. Layer " << layerIndex << endl;
		cout << "Calculated: " << net[layerIndex].prevImgLen * net[layerIndex].prevImgDepth * net[layerIndex].prevImgWid + 1 << endl;
		cout << "ACtual: " << net[layerIndex - 1].size() << endl;
		return;
	}

	//Seperate data from last layer into <depth> number of images
	vector<vector<float>> lastLayerImages;
	for (int i = 0; i < net[layerIndex].prevImgDepth; i++) {
		lastLayerImages.push_back(vector<float>());
		for (int j = 0; j < net[layerIndex].prevImgLen * net[layerIndex].prevImgWid; j++) {
			lastLayerImages.back().push_back(net[layerIndex - 1].neuron[j].activation[pipe]);
		}
	}

	int topPad = 0, rightPad = 0, bottomPad = 0, leftPad = 0;
	//If zero padding is activated, calculate how much it needs to be padded and where to add the padding
	if (net[layerIndex].zeroPad) {
		int verticalPad = net[layerIndex].filters[0].yDim - 1;
		int horizontalPad = net[layerIndex].filters[0].xDim - 1;
		bottomPad = verticalPad / 2;
		topPad = verticalPad - bottomPad;
		rightPad = horizontalPad / 2;
		leftPad = horizontalPad - rightPad;
	}
	int paddedX = net[layerIndex].prevImgLen + leftPad + rightPad, paddedy = net[layerIndex].prevImgWid + topPad + bottomPad;

	//If data is to be padded, pad it
	vector<vector<float>> paddedImages;
	for (int i = 0; i < lastLayerImages.size(); i++) {
		paddedImages.push_back(vector<float>());
		for (int j = 0; j < net[layerIndex].prevImgWid + bottomPad + topPad; j++) {
			for (int k = 0; k < net[layerIndex].prevImgLen + leftPad + rightPad; k++) {
				//If its at a boundary, push back a zero
				if (j < topPad || j >= net[layerIndex].prevImgWid + topPad || k < leftPad || k >= net[layerIndex].prevImgLen + leftPad) {
					paddedImages.back().push_back(0);
				}
				//If its not a boundary, push back the image info
				else {
					int temp = (j - topPad) * net[layerIndex].prevImgLen + (k - leftPad);
					paddedImages.back().push_back(lastLayerImages[i][temp]);
				}
			}
		}
	}

	//Convolve every filter over padded images and add respective biases
	vector<vector<float>> convolved;
	for (int i = 0; i < net[layerIndex].filters.size(); i++) {
		vector<float> sum((paddedX - net[layerIndex].filters[i].xDim + 1) * (paddedy - net[layerIndex].filters[i].yDim + 1));
		for (int j = 0; j < paddedImages.size(); j++) {
			Image temp(paddedX, paddedy, paddedImages[j]);
			vector<float> temp1 = Util::Convolve(temp, net[layerIndex].filters[i]);

			if (sum.size() != temp1.size()) {
				cout << "During convolution, my calculated image size did not match my actual convoluted size. check conv feed forward." << endl;
			}
			else {
				for (int k = 0; k < sum.size(); k++) {
					sum[k] += 1.f * temp1[k] / net[layerIndex].prevImgDepth;
				}
			}
		}
		//Add the biases to  each neuron. If this throws an error convoBias is the wrong size
		for (int j = 0; j < sum.size(); j++) {
			sum[j] += net[layerIndex].convoBias[i * sum.size() + j];
		}

		convolved.push_back(sum);
	}

	//Activate the resulting image
	for (int i = 0; i < convolved.size(); i++) {
		for (int j = 0; j < convolved[i].size(); j++) {
			convolved[i][j] = net[layerIndex].activate(convolved[i][j]);
		}
	}

	//Max pool if specified
	if (net[layerIndex].maxPoolStride != -1) {
		for (int i = 0; i < convolved.size(); i++) {
			//Make an image out of the matrix
			Image temp(paddedX - net[layerIndex].filters[0].xDim + 1, paddedy - net[layerIndex].filters[0].yDim + 1, convolved[i]);
			//Max Pool
			convolved[i] = Util::MaxPool(temp, net[layerIndex].maxPoolx, net[layerIndex].maxPooly, net[layerIndex].maxPoolStride, net[layerIndex].maxNeuronIndex[pipe][i]);
		}
	}

	//Put values into neurons
	if (convolved[0].size() * convolved.size() + 1 != net[layerIndex].size()) {
		cout << "The size I predicted this layer to have in its constructor is not the size after calculation in conv feedforward. please figure out what is wrong" << endl;
	}
	for (int i = 0; i < convolved.size(); i++) {
		for (int j = 0; j < convolved[i].size(); j++) {
			net[layerIndex].neuron[i * convolved[i].size() + j].activation[pipe] = convolved[i][j];
			net[layerIndex].neuron[i * convolved[i].size() + j].setActive(pipe, true);
		}
	}
}

void NeuralNet::ConvBackwardPass(int layerIndex, int pipe) {
	//Update Inputs
	//If the next layer isnt a conv layer, this layer neurons update using dense back pass
	if (layerIndex != net.size() - 1) {
		vector<float> gradientAtThisPipe;
		if (net[layerIndex + 1].getLayerType() != CONVO) {
			gradientAtThisPipe = GetGradientIfNextLayerDense(layerIndex, pipe);
		}
		//If the next layer is a conv layer, use filter of next layer and dActivation of next layer to calc this layer input gradient
		else {
			gradientAtThisPipe = GetGradientIfNextLayerConvo(layerIndex, pipe);
		}

		if (gradientAtThisPipe.size() != net[layerIndex].neuron.size()) {
			cout << "Gradient dimensions not equal to layer dimensions. check convsgdbackpass funciton. Layer " << layerIndex << endl;
			cout << "Gradient At this pipe: " << gradientAtThisPipe.size() << endl;
			cout << "Actual layer size: " << net[layerIndex].neuron.size() << endl;
			return;
		}
		for (int i = 0; i < gradientAtThisPipe.size(); i++) {
			net[layerIndex].neuron[i].gradient[pipe] = gradientAtThisPipe[i];
		}
	}


	//Turn input into images
	int depth = net[layerIndex].prevImgDepth;
	int layerInputWidth = net[layerIndex].prevImgWid;
	int layerInputLength = net[layerIndex].prevImgLen;
	vector<vector<float>> layerInput(depth, vector<float>(layerInputWidth * layerInputLength));

	for (int i = 0; i < depth; i++) {
		for (int j = 0; j < layerInputWidth * layerInputLength; j++) {
			layerInput[i][j] = net[layerIndex - 1].neuron[i * layerInputWidth * layerInputLength + j].activation[pipe];
		}
	}

	//Turn ouput gradients into images and multiply them by their dActivation. Note imgLen is pre max pooled length
	vector<vector<float>> layerOutput(net[layerIndex].filters.size(), vector<float>(net[layerIndex].size() / net[layerIndex].filters.size()));
	for (int i = 0; i < net[layerIndex].filters.size(); i++) {
		for (int j = 0; j < ((net[layerIndex].size() - 1) / net[layerIndex].filters.size()); j++) {
			layerOutput[i][j] = net[layerIndex].neuron[i * ((net[layerIndex].size() - 1) / net[layerIndex].filters.size()) + j].gradient[pipe] * net[layerIndex].dActivate(net[layerIndex].neuron[i * ((net[layerIndex].size() - 1) / net[layerIndex].filters.size()) + j].activation[pipe]);
		}
	}

	//Upscale output if maxPooled
	int prePoolLength = net[layerIndex].imgLen;
	int prePoolWidth = net[layerIndex].imgWid;
	int thisDepth = net[layerIndex].filters.size();
	vector<vector<float>> prePool(net[layerIndex].filters.size(), vector<float>(net[layerIndex].imgLen * net[layerIndex].imgWid));
	if (net[layerIndex].maxPoolStride == -1) {
		prePool = layerOutput;
	}
	else {
		for (int j = 0; j < thisDepth; j++) {
			if (net[layerIndex].maxNeuronIndex[pipe][j].size() != layerOutput[j].size()) {
				cout << "during backprop of the convolutional layer, the number of max neuron indexes did not match the number of max neurons. check conv sgd back pass" << endl;
				cout << "Max neuron index Array: " << net[layerIndex].maxNeuronIndex[pipe][j].size() << endl;
				cout << "Actual output of the layer: " << layerOutput[j].size() << endl;
			}
			for (int i = 0; i < net[layerIndex].maxNeuronIndex[pipe][j].size(); i++) {
				prePool[j][net[layerIndex].maxNeuronIndex[pipe][j][i]] = layerOutput[j][i];
			}
		}
	}


	//Update Filters
	vector<vector<float>> paddedImage;
	//Zero pad as necessary to get filter dimensions during convolution: inputdim - outputdim + 1 = filterdim
	//Pad = filterdim + ouputdim - 1 - inputdim
	int vertPad = net[layerIndex].filters[0].yDim + net[layerIndex].imgWid - 1 - net[layerIndex].prevImgWid;
	int horzPad = net[layerIndex].filters[0].xDim + net[layerIndex].imgLen - 1 - net[layerIndex].prevImgLen;
	int topPad = vertPad / 2, bottomPad = vertPad - topPad, leftPad = horzPad / 2, rightPad = horzPad - leftPad;
	if (topPad == 0 && leftPad == 0 && rightPad == 0 && bottomPad == 0) {
		paddedImage = layerInput;
	}
	else {
		for (int i = 0; i < layerInput.size(); i++) {
			paddedImage.push_back(vector<float>());
			for (int j = 0; j < layerInputWidth + bottomPad + topPad; j++) {
				for (int k = 0; k < layerInputLength + leftPad + rightPad; k++) {
					//If its at a boundary, push back a zero
					if (j < topPad || j >= layerInputWidth + topPad || k < leftPad || k >= layerInputLength + leftPad) {
						paddedImage[i].push_back(0);
					}
					//If its not a boundary, push back the image info
					else {
						int temp = (j - topPad) * layerInputLength + (k - leftPad);
						paddedImage[i].push_back(layerInput[i][temp]);
					}
				}
			}
		}

	}


	//Convolve
	vector<vector<float>> totalGrad(net[layerIndex].filters.size(), vector<float>(net[layerIndex].filters[0].xDim * net[layerIndex].filters[0].yDim));
	for (int i = 0; i < net[layerIndex].filters.size(); i++) {
		for (int j = 0; j < paddedImage.size(); j++) {
			Image out(net[layerIndex].imgLen, net[layerIndex].imgWid, prePool[i]);
			Image in(layerInputLength + horzPad, layerInputWidth + vertPad, paddedImage[j]);
			auto temp = Util::Convolve(in, out);
			if (totalGrad[i].size() != temp.size()) {
				cout << "Anticipated filter size is not actual filter size when calculating filter gradient. Check convSGDBackProp" << endl;
				return;
			}
			for (int k = 0; k < totalGrad[i].size(); k++) {
				totalGrad[i][k] += 1.f * temp[k] / depth;
			}

		}
	}

	//Apply update to filters
	if (totalGrad[0].size() != net[layerIndex].filters[0].xDim * net[layerIndex].filters[0].yDim) {
		cout << "The size of the calculated filter gradient isnt the size of the actual filter. check convSGDBackProp." << endl;
		return;
	}

	for (int i = 0; i < net[layerIndex].filters.size(); i++) {
		for (int j = 0; j < net[layerIndex].filters[i].xDim * net[layerIndex].filters[i].yDim; j++) {
			net[layerIndex].filters[i].gradients[j] += totalGrad[i][j];
		}
	}

	//Update Biases
	if (prePool.size() * prePool[0].size() != net[layerIndex].convoBias.size()) {
		cout << "Prepool size is not the same as bias size. check convosgdbackprop" << endl;
	}
	for (int i = 0; i < net[layerIndex].convoBias.size() - 1; i++) {
		int dep = i / (net[layerIndex].imgLen * net[layerIndex].imgWid);
		int neuronIndex = i % (net[layerIndex].imgLen * net[layerIndex].imgWid);
		net[layerIndex].convoBiasGradient[i] += prePool[dep][neuronIndex];
	}
}

//Under the assumption that we have already checked to ensure that this isnt the last layer and the next layer is dense
vector<float> NeuralNet::GetGradientIfNextLayerDense(int layerIndex, int pipe) {
	//Every neuron change is the sum of (the weight to the next neuron) * (the dActiavation of that neuron) * (Its gradient) for every neuron in the next layer except bias
	Layer nextLayer = net[layerIndex + 1];
	int thisLayerSize = nextLayer.neuron[0].weight.size();
	vector<float> result(thisLayerSize);

	//For every neuron in this layer except the bias
	for (int i = 0; i < thisLayerSize - 1; i++) {
		if (net[layerIndex].neuron[i].getActive(pipe)) {
			//For every neuron in the next layer
			for (int j = 0; j < nextLayer.size(); j++) {
				//If this neuron has the same number of weights as the previous layer has neurons(ie, all neurons except bias have weights equal to the size of the prev layer)
				if (nextLayer.neuron[j].getActive(pipe) && nextLayer.neuron[j].weight.size() == thisLayerSize) {
					result[i] += nextLayer.neuron[j].weight[i] * nextLayer.dActivate(nextLayer.neuron[j].activation[pipe]) * nextLayer.neuron[j].gradient[pipe];
				}
			}
		}
	}
	return result;
}

vector<float> NeuralNet::GetGradientIfNextLayerConvo(int layerIndex, int pipe) {
	Layer nextLayer = net[layerIndex + 1];
	//Get this layer post pool image dimensions. Work this out
	int depth = nextLayer.filters.size();
	int length = nextLayer.imgLen;
	int width = nextLayer.imgWid;
	if (nextLayer.maxPoolStride != -1) {
		float x1 = length / nextLayer.maxPoolStride;
		float y1 = width / nextLayer.maxPoolStride;
		length = (x1 > (int)x1) ? x1 + 1 : x1;
		width = (y1 > (int)y1) ? y1 + 1 : y1;
	}

	//get gradient of next layer, multiply by their dActivate
	int imageDim = 1.f * (nextLayer.size() - 1) / depth;
	vector<vector<float>> output(depth, vector<float>(imageDim));
	for (int i = 0; i < depth; i++) {
		for (int j = 0; j < imageDim; j++) {
			output[i][j] = (nextLayer.neuron[i * imageDim + j].gradient[pipe] * nextLayer.dActivate(nextLayer.neuron[i * ((nextLayer.size() - 1) / depth) + j].activation[pipe]));
		}
	}

	//Get this layer pre max pool image dimensions. Work this out
	int prePoolLength = nextLayer.imgLen;
	int prePoolWidth = nextLayer.imgWid;

	//get pre max pooled gradient matrix by placing zeros in whatever spaces that werent chosen by the max picker. ie that arent in the layers maxNeuronIndex vector
	//Am I sure ouput is in the same order as maxNeuronIndex??? Pretty sure
	//Max NeuronIndex should be a vector of vectors cuz its gonna work across different pipes
	vector<vector<float>> prePool(depth, vector<float>(prePoolLength * prePoolWidth));
	//If it wasnt max pooled then the image before pooling is equal to the output of the layer
	if (nextLayer.maxPoolStride == -1) {
		prePool = output;
	}
	else {
		for (int j = 0; j < depth; j++) {
			if (nextLayer.maxNeuronIndex[pipe][j].size() != output[j].size()) {
				cout << "during backprop of the convolutional layer, the number of max neuron indexes did not match the number of max neurons." << endl;
				cout << "Max neuron index Array: " << nextLayer.maxNeuronIndex[pipe][j].size() << endl;
				cout << "Actual output of the layer: " << output[j].size() << endl;
			}
			for (int i = 0; i < nextLayer.maxNeuronIndex[pipe][j].size(); i++) {
				prePool[j][nextLayer.maxNeuronIndex[pipe][j][i]] = output[j][i];
			}
		}
	}

	//rotate all filters 180 degrees
	vector<Image> rotatedFilter;
	for (int i = 0; i < nextLayer.filters.size(); i++) {
		Image f1 = nextLayer.filters[i];
		Util::rotate180(f1);
		rotatedFilter.push_back(f1);
	}

	vector<vector<float>> paddedImage;
	//Do whatever is necessary to get the convolution btw the output(image) and the filter(filter)  to return an image the size of the input
	//To get an input image I would need :: (Inputx + (filterx - 1)) - outputX, (inputy + (filtery-1)) - outputy :: padding
	int horzPad = nextLayer.prevImgLen + nextLayer.filters[0].xDim - 1 - nextLayer.imgLen;
	int vertPad = nextLayer.prevImgWid + nextLayer.filters[0].yDim - 1 - nextLayer.imgWid;
	int topPad = vertPad / 2, bottomPad = vertPad - topPad, leftPad = horzPad / 2, rightPad = horzPad - leftPad;
	if (topPad == 0 && leftPad == 0 && rightPad == 0 && bottomPad == 0) {
		paddedImage = prePool;
	}
	else {
		for (int i = 0; i < prePool.size(); i++) {
			paddedImage.push_back(vector<float>());
			for (int j = 0; j < prePoolWidth + bottomPad + topPad; j++) {
				for (int k = 0; k < prePoolLength + leftPad + rightPad; k++) {
					//If its at a boundary, push back a zero
					if (j < topPad || j >= prePoolWidth + topPad || k < leftPad || k >= prePoolLength + leftPad) {
						paddedImage.back().push_back(0);
					}
					//If its not a boundary, push back the image info
					else {
						int temp = (j - topPad) * prePoolLength + (k - leftPad);
						paddedImage.back().push_back(prePool[i][temp]);
					}
				}
			}
		}

	}


	vector<float> inputGradient(nextLayer.prevImgLen * nextLayer.prevImgWid * nextLayer.prevImgDepth);
	vector<float> currentGrad(nextLayer.prevImgLen * nextLayer.prevImgWid);
	vector<float> totalGrad(nextLayer.prevImgLen * nextLayer.prevImgWid);
	//Convolve all filters over the padded output. Result is input gradient
	//one point in the input image gradient is the average of all filters going over all their respective padded output gradients
	for (int i = 0; i < nextLayer.filters.size(); i++) {
		Image tmp(prePoolLength + vertPad, prePoolWidth + horzPad, paddedImage[i]);
		currentGrad = Util::Convolve(tmp, nextLayer.filters[i]);
		for (int j = 0; j < currentGrad.size(); j++) {
			totalGrad[j] += 1.f * currentGrad[j] / nextLayer.filters.size();
		}
	}


	for (int i = 0; i < inputGradient.size(); i++) {
		inputGradient[i] = totalGrad[i % (nextLayer.prevImgLen * nextLayer.prevImgWid)];
	}

	//Bias, doesnt update
	inputGradient.push_back(0);

	//Return the resulting input vector
	return inputGradient;
}

void NeuralNet::UpdateHashTables() {
	//Multithread this
	//for every layer
	for (unsigned i = 1; i < net.size(); i++) {
		vector<vector<unsigned>> weightArr;
		if (net[i].getLayerType() != CONVO) {
			//for every neuron except the bias
			for (unsigned j = 0; j < net[i].size(); j++) {
				//bias is either last neuron on every row or none existen in the last row
				if (i == net.size() - 1 || j != net[i].size() - 1) {
					vector<unsigned> intWeights;
					//for every weight except the bias weight
					for (unsigned k = 0; k < net[i - 1].size() - 1; k++) {
						intWeights.push_back(Neuron::floatToInt(net[i].neuron[j].weight[k]));
					}
					weightArr.push_back(intWeights);
				}
			}
			net[i].HashTable.UpdateTables(weightArr);
		}
	}
}

void NeuralNet::train(const vector<vector<float>>& input, const vector<vector<float>>& output) {
	//Check if input and output batches are the same size
	if (input.size() != output.size()) {
		cout << "Error: Input batch size does not match output batch size." << endl;
		return;
	}

	//Set batch size
	for (int i = 0; i < net.size(); i++) {
		net[i].maxNeuronIndex = vector<vector<vector<int>>>(input.size(), vector<vector<int>>(net[i].filters.size(), vector<int>()));
		for (int j = 0; j < net[i].size(); j++) {
			net[i].neuron[j].SetVars(input.size());
		}
	}

	float batchLoss = 0;
	clearWeightGradients();
	HashUpdateTracker();
	for (int i = 0; i < input.size(); i++) {
		//Multithread this
		feedForward(input[i], i);
		BackPropagate(output[i], i);
		batchLoss += loss;
	}
	applyWeightGradients();
	if (DEBUG)
		cout << "Batch Error: " << batchLoss / input.size() << endl;
}

void NeuralNet::trainWithOneOutput(const vector<vector<float>>& input, const vector<OneOutput>& out) {
	if (input.size() != out.size()) {
		cout << "Error: Input batch size does not match output batch size." << endl;
		return;
	}

	//Set batch size
	for (int i = 0; i < net.size(); i++) {
		net[i].maxNeuronIndex = vector<vector<vector<int>>>(input.size(), vector<vector<int>>(net[i].filters.size(), vector<int>()));
		for (int j = 0; j < net[i].size(); j++) {
			net[i].neuron[j].SetVars(input.size());
		}
	}

	clearWeightGradients();
	HashUpdateTracker();
	for (int i = 0; i < input.size(); i++) {
		//Multithread this
		feedForward(input[i], i);
		vector<float> output = getOutput(i);
		output[out[i].index] = out[i].val;
		BackPropagate(output, i);
	}
	applyWeightGradients();
}

void NeuralNet::trainTillError(const vector<vector<float>>& input, const vector<vector<float>>& output, int numOfBatches, int numOfEpochs, float targetError) {
	int batchSize = input.size() / numOfBatches;
	//Check if input and output batches are the same size
	if (input.size() != output.size()) {
		cout << "Error: Input size does not match output size. TrainTillError function" << endl;
		return;
	}

	//Set batch size
	for (int i = 0; i < net.size(); i++) {
		net[i].maxNeuronIndex = vector<vector<vector<int>>>(batchSize, vector<vector<int>>(net[i].filters.size(), vector<int>()));
		for (int j = 0; j < net[i].size(); j++) {
			net[i].neuron[j].SetVars(batchSize);
		}
	}

	for (int i = 0; i < numOfEpochs; i++) {
		for (int j = 0; j < numOfBatches; j++) {
			float batchLoss = 0.f;
			clearWeightGradients();
			HashUpdateTracker();
			for (int k = 0; k < batchSize; k++) {
				if (j * batchSize + k > input.size()) break;
				//Multithread this
				feedForward(input[j * batchSize + k], k);
				BackPropagate(output[j * batchSize + k], k);
				batchLoss += loss;
			}
			applyWeightGradients();
			if (DEBUG) cout << "Batch Error: " << batchLoss / batchSize << endl;
			if (batchLoss / batchSize < targetError) return;
		}
	}
}

void NeuralNet::HashUpdateTracker() {
	iter++;
	if (iter == nextUpdate) {
		t++;
		UpdateHashTables();
		nextUpdate += baseT * exp(lambda * (t - 1));
	}
}

void NeuralNet::DebugWeights() {
	for (int i = 1; i < net.size(); i++) {
		for (int j = 0; j < net[i].size(); j++) {
			PrintVec(net[i].neuron[j].weight);
		}
		cout << endl;
	}
}

void NeuralNet::DebugWeights(int layer) {
	int i = layer;
	for (int j = 0; j < net[i].size(); j++) {
		PrintVec(net[i].neuron[j].weight);
	}
	cout << endl;
}

void NeuralNet::save(string filename) {
	//Open file
	ofstream outFile(filename);

	//write the max weight used for hashing
	outFile << Neuron::maxW << endl;

	//Write number of layers
	outFile << net.size() << endl;

	//Write layer Configs
	for (int i = 0; i < net.size(); i++) {
		int netSize = (i == net.size() - 1) ? net[i].size() : net[i].size() - 1;
		if (net[i].getLayerType() == DENSE) {
			outFile << net[i].getLayerType() << " " << netSize << " ";
			outFile << net[i].getActivationFunction() << " " << net[i].bits << " ";
			outFile << net[i].tables << " " << net[i].neuronLimit << endl;
		}
		else if (net[i].getLayerType() == CONVO) {
			outFile << net[i].getLayerType() << " " << net[i].filters[0].xDim << " ";
			outFile << net[i].filters[0].yDim << " " << net[i].filters.size() << " ";
			outFile << net[i].getActivationFunction() << " " << net[i].prevImgLen << " ";
			outFile << net[i].prevImgWid << " " << net[i].prevImgDepth << " ";
			outFile << net[i].zeroPad << " " << net[i].maxPoolStride << endl;
		}
	}

	//Write layer values
	for (int i = 1; i < net.size(); i++) {
		if (net[i].getLayerType() == DENSE) {
			//Retrieve and store table hashes
			vector<vector<unsigned>> tableHashes = net[i].HashTable.getTableHashes();
			for (int j = 0; j < tableHashes.size(); j++) {
				for (int k = 0; k < tableHashes[j].size(); k++) {
					outFile << tableHashes[j][k];
					if (k != tableHashes[j].size() - 1) {
						//outFile << " ";
					}
				}
				outFile << endl;
			}
			//Retrieve and store weights
			for (int j = 0; j < net[i].size(); j++) {
				if (i == net.size() - 1 || !(j == net[i].size() - 1)) {
					for (int k = 0; k < net[i - 1].size(); k++) {
						outFile << net[i].neuron[j].weight[k];
						if (k != net[i - 1].size() - 1) outFile << " ";
					}
					outFile << endl;
				}
			}
			outFile << endl;
		}
		else if (net[i].getLayerType() == CONVO) {
			//Retrieve and store filters
			for (int j = 0; j < net[i].filters.size(); j++) {
				for (int k = 0; k < net[i].filters[j].val.size(); k++) {
					outFile << net[i].filters[j].val[k] << " ";
				}
				outFile << endl;
			}

			for (int j = 0; j < net[i].convoBias.size(); j++) {
				outFile << net[i].convoBias[j] << " ";
			}
			outFile << endl;
		}
	}

	//Close file
	outFile.close();
}

bool NeuralNet::load(string fileName) {
	fstream file;
	file.open(fileName);
	if (file.fail()) {
		return false;

	}
	else {
		LoadCurrNetVersion(ifstream(fileName));
		return true;
	}
}

void NeuralNet::LoadCurrNetVersion(ifstream rd) {
	//read max weight used for hashing
	float max;
	rd >> max;
	Neuron::maxW = max;

	//Read number of layers
	int netSize;
	vector<Layer> layerArray;
	rd >> netSize;

	//Read layer config
	int layerType, layerSize, activeFunc, bits, tables, neuronLim;
	int filterx, filtery, filternum, prevImgLen, prevImgWid, prevImgDepth, zPad, maxPoolStride;
	for (int i = 0; i < netSize; i++) {
		rd >> layerType;
		if (layerType == (int)DENSE) {
			rd >> layerSize >> activeFunc >> bits >> tables >> neuronLim;
			layerArray.push_back(Layer((LayerType)layerType, layerSize, (ActivationFunction)activeFunc, bits, tables, neuronLim));
		}
		else if (layerType == (int)CONVO) {
			rd >> filterx >> filtery >> filternum >> activeFunc >> prevImgLen >> prevImgWid >> prevImgDepth >> zPad >> maxPoolStride;
			layerArray.push_back(Util::Convo(filterx, filtery, filternum, (ActivationFunction)activeFunc, prevImgLen, prevImgWid, prevImgDepth, zPad, maxPoolStride));
		}
		else {
			cout << "Did not recognize loaded layer. check load function." << endl;
		}
	}

	startNetwork(layerArray);

	for (int i = 1; i < net.size(); i++) {
		if (net[i].getLayerType() == DENSE) {
			//Read all hashes
			vector<vector<unsigned>> tableHashes;
			for (int j = 0; j < net[i].tables; j++) {
				vector<unsigned> hash;
				for (int k = 0; k < net[i].bits; k++) {
					int g;
					rd >> g;
					hash.push_back(g);
				}
				tableHashes.push_back(hash);
			}
			net[i].HashTable.setTableHashes(tableHashes);

			//Read all weights
			vector<vector<unsigned>> weightArray;
			for (int j = 0; j < net[i].size(); j++) {
				if (i == net.size() - 1 || j != net[i].size() - 1) {
					vector<float> weight;
					vector<unsigned> intWeight;
					for (int k = 0; k < net[i - 1].size(); k++) {
						float h;
						rd >> h;
						weight.push_back(h);
						//Don't add the bias weight to the hash table
						if (k + 1 != net[i - 1].size())
							intWeight.push_back(Neuron::floatToInt(weight.back()));
					}
					net[i].neuron[j] = (Neuron(weight));
					weightArray.push_back(intWeight);
				}
				else {
					net[i].neuron.push_back(Neuron());
				}
			}
			net[i].HashTable.Hash(weightArray);
		}
		else if (net[i].getLayerType() == CONVO) {
			for (int k = 0; k < net[i].filters.size(); k++) {
				vector<float> img;
				float tmp;
				for (int j = 0; j < net[i].filters[0].xDim * net[i].filters[0].yDim; j++) {
					rd >> tmp;
					img.push_back(tmp);
				}
				Image filt(net[i].filters[0].xDim, net[i].filters[0].yDim, img);
				net[i].filters[k] = filt;
			}
			for (int k = 0; k < net[i].convoBias.size(); k++) {
				float tmp;
				rd >> tmp;
				net[i].convoBias[k] = tmp;
			}
		}
	}
}

float NeuralNet::getMaxOutput() {
	int pipe = 0;
	float max = net[net.size() - 1].neuron[0].activation[pipe];
	for (int i = 1; i < net[net.size() - 1].neuron.size(); i++) {
		if (net[net.size() - 1].neuron[i].activation[pipe] > max) {
			max = net[net.size() - 1].neuron[i].activation[pipe];
		}
	}
	return max;
}

int NeuralNet::getMaxOutputIndex() {
	int pipe = 0;
	float max = 0;
	for (int i = 1; i < net[net.size() - 1].neuron.size(); i++) {
		if (net[net.size() - 1].neuron[i].activation[pipe] > net[net.size() - 1].neuron[max].activation[pipe]) {
			max = i;
		}
	}
	return max;
}

vector<float> NeuralNet::getOutput() {
	return getOutput(0);
}

vector<float> NeuralNet::getOutput(int pipe) {
	return net[net.size() - 1].inputAt(pipe);
}

void NeuralNet::operator=(const NeuralNet& obj) {
	this->net = obj.net;
}

vector<int> NeuralNet::getConvLayerImgSize(int layerIndex) {
	int x = net[layerIndex].imgLen;
	int y = net[layerIndex].imgWid;
	if (net[layerIndex].maxPoolStride != -1) {
		float tempX = 1.f * x / net[layerIndex].maxPoolStride;
		float tempY = 1.f * y / net[layerIndex].maxPoolStride;
		x = (tempX > (int)tempX) ? tempX + 1 : tempX;
		y = (tempY > (int)tempY) ? tempY + 1 : tempY;
	}
	return { x, y };
}

int NeuralNet::getConvLayerFilterSize(int layerIndex) {
	return net[layerIndex].filters.size();
}

int NeuralNet::getLayerSize(int layerIndex) {
	return net[layerIndex].size();
}

void NeuralNet::clearWeightGradients() {
	for (int i = 1; i < net.size(); i++) {
		for (int j = 0; j < net[i].size(); j++) {
			//If the weights connecting to this neuron is the same number as the number of neurons in the previous layer(ie, not the bias neuron)
			if (net[i].neuron[j].weight.size() == net[i - 1].size()) {
				for (int k = 0; k < net[i].neuron[j].weight.size(); k++) {
					for (int l = 0; l < net[i].neuron[j].activation.size(); l++) {
						net[i].neuron[j].weightGradient[k][l] = 0;
					}
				}
			}
		}
		for (int j = 0; j < net[i].convoBiasGradient.size(); j++) {
			net[i].convoBiasGradient[j] = 0;
		}
		for (int j = 0; j < net[i].filters.size(); j++) {
			for (int k = 0; k < net[i].filters[j].gradients.size(); k++) {
				net[i].filters[j].gradients[k] = 0;
			}
		}
	}
}

void NeuralNet::applyWeightGradients() {
	float batchsize = net[0].neuron[0].activation.size();
	float gr = 1.f * growthRate / batchsize;
	float B = adaptiveLearningRateHyperparameter;
	float A = momentumHyperparameter;
	float e = 0.001;//small number added to adaptive learningrate so I wont have any divide by zeros
	for (int i = 1; i < net.size(); i++) {
		for (int j = 0; j < net[i].size(); j++) {
			//If the weights connecting to this neuron is the same number as the number of neurons in the previous layer(ie, not the bias neuron)
			if (net[i].neuron[j].weight.size() == net[i - 1].size()) {
				for (int k = 0; k < net[i].neuron[j].weight.size(); k++) {
					float totalGradient = 0;
					for (int l = 0; l < net[i].neuron[j].activation.size(); l++) {
						totalGradient += net[i].neuron[j].weightGradient[k][l];
					}

					net[i].neuron[j].adaptiveLearningRate[k] = B * net[i].neuron[j].adaptiveLearningRate[k] + (1 - B) * (totalGradient * totalGradient);
					net[i].neuron[j].momentum[k] = A * net[i].neuron[j].momentum[k] + (1 - A) * totalGradient;
					net[i].neuron[j].weight[k] -= gr / (sqrt(net[i].neuron[j].adaptiveLearningRate[k]) + e) * net[i].neuron[j].momentum[k];
				}
			}
		}
		for (int j = 0; j < net[i].convoBiasGradient.size(); j++) {
			net[i].convoBiasAdaptiveLearningRate[j] = B * net[i].convoBiasAdaptiveLearningRate[j] + (1 - B) * (net[i].convoBiasGradient[j] * net[i].convoBiasGradient[j]);
			net[i].convoBiasMomentum[j] = A * net[i].convoBiasMomentum[j] + (1 - A) * net[i].convoBiasGradient[j];
			net[i].convoBias[j] -= gr / (sqrt(net[i].convoBiasAdaptiveLearningRate[j]) + e) * net[i].convoBiasMomentum[j];
		}
		for (int j = 0; j < net[i].filters.size(); j++) {
			for (int k = 0; k < net[i].filters[j].gradients.size(); k++) {
				net[i].filters[j].valAdaptiveLearningRate[k] = B * net[i].filters[j].valAdaptiveLearningRate[k] + (1 - B) * (net[i].filters[j].gradients[k] * net[i].filters[j].gradients[k]);
				net[i].filters[j].valMomentum[k] = A * net[i].filters[j].valMomentum[k] + (1 - A) * net[i].filters[j].gradients[k];
				net[i].filters[j].val[k] -= gr / (sqrt(net[i].filters[j].valAdaptiveLearningRate[k]) + e) * net[i].filters[j].valMomentum[k];
			}
		}
	}
}