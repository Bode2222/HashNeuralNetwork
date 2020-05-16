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

float NeuralNet::Cost(float input, float target) {
	//If im using softmax to classify return cross entropy loss
	if (net[net.size() - 1].getActivationFunction() == SOFTMAX) {
		float temp = -target * log(input);
		return temp;
	}
	//else return mean squared error loss
	return (input - target) * (input - target);
}

float NeuralNet::CostDerivative(float myOutput, float target) {
	//If im using softmax to classify return cross entropy loss derivative
	if (net[net.size() - 1].getActivationFunction() == SOFTMAX) {
		return (target == 1) ? (myOutput - target) : myOutput;
	}
	//else return mean squared error derivative
	float result = (myOutput - target) * net[net.size() - 1].dActivate(myOutput);
	return result;
}

float NeuralNet::getError() {
	return totalError;
}

/*When instantiating the network, instantiate the layers
*then reach into them and individually assign each neuron random weights
*After which hash all neurons of each layer into all tables of each layer
*based on their weights*/
NeuralNet::NeuralNet(vector<Layer>& layout) {
	net = layout;
	//Remove the bais from the last layer
	net[net.size() - 1].setSize(net[net.size() - 1].size() - 1);

	//for every layer
	for (int i = 0; i < net.size(); i++) {
		vector<vector<unsigned>> weightArray;
		//for every neuron.
		for (int j = 0; j < net[i].size(); j++) {
			//!(If its the input layer, or its the last neuron and its not the last layer)
			if (!(i == 0) && !(j == net[i].size() - 1 && i < net.size() - 1)) {
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
			//no need for weights on the input layer or the bias node and batchsize is set when training
			else {
				net[i].neuron.push_back(Neuron());
			}
		}
		//dont hash input layer
		if (i != 0) net[i].HashTable.Hash(weightArray);
	}
}

void NeuralNet::feedForward(vector<float> input, int pipe) {
	//if input doesnt match the input layer, throw error
	if (input.size() != net[0].neuron.size() - 1) {
		cout << "Our layer size is " << net[0].neuron.size() - 1 << endl;
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
	}
}

void NeuralNet::feedForward(const vector<float>& input) {
	//Set batch size
	if (net[0].neuron[0].activation.size() == 0) {
		for (int i = 0; i < net.size(); i++) {
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
	totalError = 0;
	for (unsigned i = 0; i < output.size(); i++) {
		if (net[outLayIndex].neuron[i].getActive(pipe)) {
			net[outLayIndex].neuron[i].gradient[pipe] = CostDerivative(net[outLayIndex].neuron[i].activation[pipe], output[i]);
			totalError += Cost(net[outLayIndex].neuron[i].activation[pipe], output[i]);
		}
	}

	//For every layer except the input. update weights and neurons
	for (int i = outLayIndex; i > 0; i--) {
		if (net[i].getLayerType() == DENSE) DenseSGDBackPass(i, pipe);
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

void NeuralNet::DenseSGDBackPass(int layerIndex, int pipe) {
	//Where do I update the weights for the last layer?
	//for every neuron
	for (int i = 0; i < net[layerIndex].size(); i++) {
		//if the neuron is active
		if (net[layerIndex].neuron[i].getActive(pipe)) {
			//Update the neuron
			//for every weight leaving it (Updating the neuron gradient)
			if (layerIndex != net.size() - 1) {//&& !(layerIndex != net.size() - 1 && i == net[layerIndex].size() - 1)) {//except the last layer and the bias
				for (int j = 0; j < net[layerIndex + 1].size(); j++) {
					//if the weight is going to an active neuron
					//And that neuron is not the bias
					//this activates if we are pointing to the last layer or if j isnt the last neuron
					if (layerIndex + 1 == net[layerIndex].size() - 1 || j != net[layerIndex + 1].size() - 1) {
						if (net[layerIndex + 1].neuron[j].getActive(pipe)) {
							Neuron* nextNeuron = &net[layerIndex + 1].neuron[j];
							Layer* nextLayer = &net[layerIndex + 1];
							net[layerIndex].neuron[i].gradient[pipe] += nextNeuron->weight[i] * nextLayer->dActivate(nextNeuron->activation[pipe]) * nextNeuron->gradient[pipe];
						}
					}
				}
			}
			//cout << "change for neuron " << layerIndex << ", " << i << " is " << net[layerIndex].neuron[i].gradient[pipe] << endl;
			//for every weight that is connected to it (Updating the weights)

			//cout << i << ": ";
			//PrintVec<float>(net[layerIndex].neuron[i].weight);

			//Update the weights leading into that neuron
			//Changed i == to i != and removed the ! from the if statement
			if (!(layerIndex != net.size() - 1 && i == net[layerIndex].size() - 1)) {//except bias
				for (int j = 0; j < net[layerIndex - 1].size(); j++) {
					//if the weight is coming from an active neuron
					if (net[layerIndex - 1].neuron[j].getActive(pipe)) {
						Neuron* prevNeuron = &net[layerIndex - 1].neuron[j];
						Neuron* curNeuron = &net[layerIndex].neuron[i];
						//weight update is divided by batch size and diminished by growthRate
						net[layerIndex].neuron[i].weight[j] -= ((prevNeuron->activation[pipe] * net[layerIndex].dActivate(curNeuron->activation[pipe]) * curNeuron->gradient[pipe]) / curNeuron->activation.size()) * growthRate;

						//cout << net[layerIndex].dActivate(curNeuron->activation[pipe]) << endl;
						//cout << "Weight: " << net[layerIndex].neuron[i].weight[j] << " Change " << j << ", " << i << ": " << ((prevNeuron->activation[pipe] * net[layerIndex].dActivate(curNeuron->activation[pipe]) * curNeuron->gradient[pipe]) / curNeuron->activation.size()) * growthRate << endl;
					}
				}
			}
		}
	}
}

void NeuralNet::UpdateHashTables() {
	//Multithread this
	//for every layer
	for (unsigned i = 1; i < net.size(); i++) {
		vector<vector<unsigned>> weightArr;
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

void NeuralNet::train(const vector<vector<float>>& input, const vector<vector<float>>& output) {
	//Check if input and output batches are the same size
	if (input.size() != output.size()) {
		cout << "Error: Input batch size does not match output batch size." << endl;
		return;
	}

	//Set batch size
	for (int i = 0; i < net.size(); i++) {
		for (int j = 0; j < net[i].size(); j++) {
			net[i].neuron[j].SetVars(input.size());
		}
	}

	HashUpdateTracker();
	for (int i = 0; i < input.size(); i++) {
		//Multithread this
		feedForward(input[i], i);
		BackPropagate(output[i], i);
	}
}

void NeuralNet::trainWithOneOutput(const vector<vector<float>>& input, const vector<OneOutput>& out) {
	if (input.size() != out.size()) {
		cout << "Error: Input batch size does not match output batch size." << endl;
		return;
	}

	//Set batch size
	for (int i = 0; i < net.size(); i++) {
		for (int j = 0; j < net[i].size(); j++) {
			net[i].neuron[j].SetVars(input.size());
		}
	}

	HashUpdateTracker();
	for (int i = 0; i < input.size(); i++) {
		//Multithread this
		feedForward(input[i], i);
		vector<float> output = getOutput(i);
		output[out[i].index] = out[i].val;
		BackPropagate(output, i);
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
		int netSize = (i == net[i].size() - 1) ? net[i].size() : net[i].size() - 1;
		outFile << net[i].getLayerType() << " " << net[i].size() << " ";
		outFile << net[i].getActivationFunction() << " " << net[i].bits << " ";
		outFile << net[i].tables << " " << net[i].neuronLimit << endl;
	}

	//Write all weights and hash tables
	for (int i = 1; i < net.size(); i++) {
		//Retrieve and store table hashes
		vector<vector<unsigned>> tableHashes = net[i].HashTable.getTableHashes();
		for (int j = 0; j < tableHashes.size(); j++) {
			for (int k = 0; k < tableHashes[j].size(); k++) {
				outFile << tableHashes[j][k];
				if (k != tableHashes[j].size() - 1) outFile << " ";
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
	int num;
	vector<Layer> layout;
	rd >> num;

	//Read layer config
	int a, b, c, d, e, f;
	for (int i = 0; i < num; i++) {
		rd >> a >> b >> c >> d >> e >> f;
		if (i < num - 1) b--;
		layout.push_back(Layer((LayerType)a, b, (ActivationFunction)c, d, e, f));
	}

	net = layout;
	net[net.size() - 1].setSize(net[net.size() - 1].size() - 1);

	//Create Neurons of the first layer
	for (int i = 0; i < net[0].size(); i++) {
		net[0].neuron.push_back(Neuron());
	}
	for (int i = 1; i < net.size(); i++) {
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
			if (i == net.size() - 1 || !(j == net[i].size() - 1)) {
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
				net[i].neuron.push_back(Neuron(weight));
				weightArray.push_back(intWeight);
			}
			else {
				net[i].neuron.push_back(Neuron());
			}
		}
		if (i != 0) net[i].HashTable.Hash(weightArray);
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
