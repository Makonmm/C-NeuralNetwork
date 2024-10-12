
#include <cstdlib>
#include <random>
#include <cmath>
#include <iostream>
#include <exception>
#include <string>
#include <vector>
#include <assert.h>


using namespace std;

class Neuron
{

private:

	
	int neuronInputListCount;

	
	Neuron* inputNeurons;

	double activation, activationNudgeSum;

	double* weights, * weightsMomentum;

	double bias, biasMomentum;

	double momentumRetention;

protected:

	double getActivationFunctionInput() const
	{
		double sumOfProduct = 0;
		for (auto i = 0; i < neuronInputListCount; i++)
		{
			sumOfProduct += weights[i] * inputNeurons[i].getActivation();
		}

		return sumOfProduct + bias;
	}


	double getActivationNudgeSum() const
	{
		return activationNudgeSum;
	}

	virtual double getActivationRespectiveDerivation(const int inputNeuronIndex) const
	{
		assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

		return activationNudgeSum * weights[inputNeuronIndex];
	}


	virtual double getWeightRespectiveDerivation(const int inputNeuronIndex) const
	{
		assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

		return activationNudgeSum * inputNeurons[inputNeuronIndex].getActivation();
	}

	
	virtual double getBiasRespectiveDerivation() const
	{
		assert(neuronInputListCount >= 0);

		return activationNudgeSum * 1.0;
	}

	
	void nudgeActivation(double nudge)
	{
		activationNudgeSum += nudge;
	}

public:

	
	Neuron() : weights(nullptr), weightsMomentum(nullptr), inputNeurons(nullptr)
	{
		this -> neuronInputListCount = 0;
		this -> momentumRetention = 0;

		bias = biasMomentum = 0.0;

		activation = activationNudgeSum = 0.0;
	}

	Neuron(int neuronInputListCount, Neuron* inputNeurons, double momentumRetention = 0.0)
	{
		this -> neuronInputListCount = neuronInputListCount;
		this -> inputNeurons = inputNeurons;
		this -> momentumRetention = momentumRetention;

		
		random_device randomDevice{};
		mt19937 generator{ randomDevice() };
		normal_distribution<double> randomGaussianDistributor{ 0.0, sqrt(2 / (double)neuronInputListCount) };

		weights = new double[neuronInputListCount];
		if (weights == nullptr) throw bad_alloc();
		for (auto i = 0; i < neuronInputListCount; i++)
		{
			weights[i] = randomGaussianDistributor(generator);
		}

		weightsMomentum = new double[neuronInputListCount]();
		if (weightsMomentum == nullptr) throw bad_alloc();

		bias = biasMomentum = 0.0;

		activation = activationNudgeSum = 0.0;
	}

	Neuron(int neuronInputListCount, Neuron* inputNeurons, vector<double> weightValues, double biasValue, double momentumRetention = 0.0)
	{
		this -> neuronInputListCount = neuronInputListCount;
		this -> inputNeurons = inputNeurons;
		this -> momentumRetention = momentumRetention;


		weights = new double[neuronInputListCount];
		if (weights == nullptr) throw bad_alloc();
		for (auto i = 0; i < neuronInputListCount; i++)
			weights[i] = weightValues[i];

		weightsMomentum = new double[neuronInputListCount]();
		if (weightsMomentum == nullptr) throw bad_alloc();

		bias = biasValue;
		biasMomentum = 0.0;

		activation = activationNudgeSum = 0.0;
	}

	Neuron(const Neuron& original)
	{
		neuronInputListCount = original.neuronInputListCount;
		inputNeurons = original.inputNeurons;
		activation = original.activation;
		activationNudgeSum = original.activationNudgeSum;
		bias = original.bias;
		biasMomentum = original.biasMomentum;
		momentumRetention = original.momentumRetention;

		weights = new double[neuronInputListCount];
		if (weights == nullptr) throw bad_alloc();
		for (auto i = 0; i < neuronInputListCount; i++)
			weights[i] = original.weights[i];

		weightsMomentum = new double[neuronInputListCount];
		if (weightsMomentum == nullptr) throw bad_alloc();
		for (auto i = 0; i < neuronInputListCount; i++)
			weightsMomentum[i] = original.weightsMomentum[i];
	}

	Neuron& operator=(const Neuron& original)
	{
		neuronInputListCount = original.neuronInputListCount;
		inputNeurons = original.inputNeurons;
		activation = original.activation;
		activationNudgeSum = original.activationNudgeSum;
		bias = original.bias;
		biasMomentum = original.biasMomentum;
		momentumRetention = original.momentumRetention;

		weights = new double[neuronInputListCount];
		if (weights == nullptr) throw bad_alloc();
		for (auto i = 0; i < neuronInputListCount; i++)
			weights[i] = original.weights[i];

		weightsMomentum = new double[neuronInputListCount];
		if (weightsMomentum == nullptr) throw bad_alloc();
		for (auto i = 0; i < neuronInputListCount; i++)
			weightsMomentum[i] = original.weightsMomentum[i];

		return *this;
	}

	~Neuron()
	{
		inputNeurons = nullptr;

		delete[] weights;
		delete[] weightsMomentum;
	}

	virtual void activate(const double input = 0.0)
	{
		if (neuronInputListCount > 0)
		{
			activation = getActivationFunctionInput();
		}
		else
		{
			activation = input;
		}

	}

	void setError(double cost)
	{
		activationNudgeSum = cost;
	}

	void injectInputRespectiveCostDerivation() const
	{
		for (auto i = 0; i < neuronInputListCount; i++)
		{
			inputNeurons[i].nudgeActivation(getActivationRespectiveDerivation(i));
		}
	}

	void updateWeights(int batchSize, double learningRate)
	{
		for (auto i = 0; i < neuronInputListCount; i++)
		{
			weightsMomentum[i] = momentumRetention * weightsMomentum[i] - (getWeightRespectiveDerivation(i) / batchSize) * learningRate;
			weights[i] += weightsMomentum[i];
		}
	}

	
	void updateBias(int batchSize, double learningRate)
	{
		biasMomentum = momentumRetention * biasMomentum - (getBiasRespectiveDerivation() / batchSize) * learningRate;
		bias += biasMomentum;
	}

	void resetNudges()
	{
		activationNudgeSum = 0.0;
	}


	int getInputCount() const
	{
		return neuronInputListCount;
	}

	double getActivation() const
	{
		return activation;
	}

	double getWeight(int inputNeuronIndex) const
	{
		assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

		return weights[inputNeuronIndex];
	}

	double getBias() const
	{
		return bias;
	}

	virtual string getNeuronType()
	{
		return getInputCount() == 0 ? "Input" : "Linear";
	}

};

class NeuralLayer
{

protected:

	int neuronArrayLength, neuronArrayWidth;

	Neuron* neurons;

	NeuralLayer* previousLayer;

	void setError(double costArray[])
	{
		if (costArray != nullptr)
		{
			for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
				neurons[i].setError(costArray[i]);
		}
	}

	void injectErrorBackwards()
	{
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
			neurons[i].injectInputRespectiveCostDerivation();
	}

	void updateParameters(int batchSize, double learningRate)
	{
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i].updateWeights(batchSize, learningRate);

			neurons[i].updateBias(batchSize, learningRate);
		}
	}


	void clearNudges()
	{
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
			neurons[i].resetNudges();
	}

public:
	
	NeuralLayer()
	{
		neurons = nullptr;
		neuronArrayLength = 0;
		neuronArrayWidth = 0;
		previousLayer = nullptr;
	}

	NeuralLayer(int inputLength, int inputWidth) : neuronArrayLength(inputLength), neuronArrayWidth(inputWidth), previousLayer(nullptr)
	{
		neurons = new Neuron[inputLength * inputWidth];
		if (neurons == nullptr) throw bad_alloc();

		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i] = Neuron();
		}
	}

	NeuralLayer(int neuronCount, NeuralLayer* inputLayer, double momentumRetention = 0.0)
	{
		neuronArrayLength = neuronCount;
		neuronArrayWidth = 1;
		previousLayer = inputLayer;

		int inputNeuronCount = previousLayer->getNeuronArrayCount();
		Neuron* inputNeurons = previousLayer->getNeurons();
		neurons = new Neuron[neuronCount];
		if (neurons == nullptr) throw bad_alloc();

		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i] = Neuron(inputNeuronCount, inputNeurons, momentumRetention);
		}

	}

	NeuralLayer(int neuronCount, NeuralLayer* inputLayer, double momentumRetention, vector<vector<double>> weightValues, vector<double> biasValues)
	{
		neuronArrayLength = neuronCount;
		neuronArrayWidth = 1;
		previousLayer = inputLayer;

		int inputNeuronCount = previousLayer->getNeuronArrayCount();
		Neuron* inputNeurons = previousLayer->getNeurons();
		neurons = new Neuron[neuronCount];
		if (neurons == nullptr) throw bad_alloc();

		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i] = Neuron(inputNeuronCount, inputNeurons, weightValues[i], biasValues[i], momentumRetention);
		}

	}

	NeuralLayer(const NeuralLayer& original)
	{
		neuronArrayLength = original.neuronArrayLength;
		neuronArrayWidth = original.neuronArrayWidth;
		previousLayer = original.previousLayer;

		neurons = new Neuron[neuronArrayLength * neuronArrayWidth];
		if (neurons == nullptr) throw bad_alloc();

		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i] = Neuron(original.neurons[i]);
		}
	}

	NeuralLayer& operator=(const NeuralLayer& original)
	{
		neuronArrayLength = original.neuronArrayLength;
		neuronArrayWidth = original.neuronArrayWidth;
		previousLayer = original.previousLayer;

		neurons = new Neuron[neuronArrayLength * neuronArrayWidth];
		if (neurons == nullptr) throw bad_alloc();

		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i] = Neuron(original.neurons[i]);
		}

		return (*this);
	}

	~NeuralLayer()
	{
		delete[] neurons;

		previousLayer = nullptr;
	}

	void propagateForward(double inputValues[] = nullptr)
	{
		if (previousLayer == nullptr)
		{
			for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
			{
				neurons[i].activate(inputValues[i]);
			}
		}

		else
		{
			for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
			{
				neurons[i].activate();
			}
		}

		clearNudges();
	}

	void propagateBackward(int batchSize, double learningRate, double* costArray = nullptr)
	{
		setError(costArray);

		injectErrorBackwards();

		updateParameters(batchSize, learningRate);
	}

	int getNeuronArrayLength() const
	{
		return neuronArrayLength;
	}

	int getNeuronArrayWidth() const
	{
		return neuronArrayWidth;
	}

	int getNeuronArrayCount() const
	{
		return getNeuronArrayLength() * getNeuronArrayWidth();
	}

	Neuron* getNeurons() const
	{
		return neurons;
	}

	NeuralLayer* getPreviousLayer() const
	{
		return previousLayer;
	}

	vector<double> getNeuronActivations() const
	{
		vector<double> neuronActivations;

		for (auto i = 0; i < getNeuronArrayCount(); i++)
		{
			neuronActivations.push_back(getNeurons()[i].getActivation());
		}

		return neuronActivations;
	}

	virtual string getNeuralLayerType() const
	{
		return previousLayer == nullptr ? "Input" : neurons[0].getNeuronType();
	}
};

struct layerCreationInfo
{
	string type;
	int neuronCount;
	double momentumRetention;
};

struct layerLoadInfo
{
	string type;
	int neuronCount;
	double momentumRetention;
	vector<vector<double>> weightsOfNeurons;
	vector<double> biasOfNeurons;
};

class NeuralNetwork
{

private:

	int layerCount;

	int inputLength, inputWidth;

	int outputCount;

	NeuralLayer* neuralLayers;

	double learningRate;

	int batchSize;

public:
	
	NeuralNetwork(int layerCount, int inputLength, int inputWidth, int outputCount, double learningRate, int batchSize, layerCreationInfo* layerDetails)
	{
		this -> layerCount = layerCount;
		this -> inputLength = inputLength;
		this -> inputWidth = inputWidth;
		this -> outputCount = outputCount;
		this -> learningRate = learningRate;
		this -> batchSize = batchSize;

		neuralLayers = new NeuralLayer[layerCount];
		if (neuralLayers == nullptr) throw bad_alloc();
		neuralLayers[0] = NeuralLayer(inputLength, inputWidth);

		for (auto i = 1; i < layerCount; i++)
		{

			if (false)
			{
			}
			else
			{
				neuralLayers[i] = NeuralLayer(layerDetails[i].neuronCount, &neuralLayers[i - 1], layerDetails[i].momentumRetention);
			}
		}
	}

	vector<double> getOutputs()
	{
		return neuralLayers[layerCount - 1].getNeuronActivations();
	}

	void propagateForwards(double* inputMatrix)
	{
		neuralLayers[0].propagateForward(inputMatrix);

		for (auto i = 1; i < layerCount; i++)
		{
			neuralLayers[i].propagateForward();
		}
	}

	void propagateBackwards(double* costArray)
	{
		neuralLayers[layerCount - 1].propagateBackward(batchSize, learningRate, costArray);

		for (auto i = layerCount - 2; i > 0; i--)
		{
			neuralLayers[i].propagateBackward(batchSize, learningRate);
		}
	}

	void updateBatchSize(int newBatchSize)
	{
		batchSize = newBatchSize;
	}

	void updateLearningRate(int newLearningRate)
	{
		learningRate = newLearningRate;
	}

};


int main()
{
	
	int numberOfLayers, inputLength, inputWidth, outputCount, batchSize;

	cout << "What is the length of the neural network? ";
	cin >> inputLength;
	cout << endl;

	inputWidth = 1;

	cout << "How many outputs this neural network will have? ";
	cin >> outputCount;
	cout << endl;

	cout << "How many layers this neural network will have? ";
	cin >> numberOfLayers;
	layerCreationInfo* layerDetails = new layerCreationInfo[numberOfLayers];
	cout << endl;

	batchSize = 1;

	layerDetails[0].type = "1";
	layerDetails[0].neuronCount = inputLength * inputWidth;

	layerDetails[0].momentumRetention = 0;

	for (int i = 1; i < numberOfLayers; i++)
	{
		cout << endl << "Define neural network: " << i + 1 << ":\n";

		cout << "Activation type: ";
		cin >> layerDetails[i].type;
		cout << endl;

		if (i + 1 < numberOfLayers)
		{
			cout << "Neuron count: ";
			cin >> layerDetails[i].neuronCount;
			cout << endl;
		}
		else
		{
			layerDetails[i].neuronCount = outputCount;
		}

		cout << "Momentum retention: ";
		cin >> layerDetails[i].momentumRetention;
		layerDetails[i].momentumRetention = 0;
		cout << endl;
	}

	NeuralNetwork network = NeuralNetwork(numberOfLayers, inputLength, inputWidth, outputCount, 0.0001, batchSize, layerDetails);

	double* inputGrid = new double[inputLength * inputWidth];
	for (auto i = 0; i < inputLength * inputWidth; i++)
	{
		inputGrid[i] = 15;
	}

	network.propagateForwards(inputGrid);

	auto outputVector = network.getOutputs();
	for (vector<double>::iterator it = outputVector.begin(); it < outputVector.end(); it++)
	{
		cout << (*it) << " ";
	}

	double* errorVector = new double[outputCount];
	for (auto i = 0; i < outputCount; i++)
	{
		errorVector[i] = (2 / outputCount) * (20 - network.getOutputs()[i]) * (-1);
	} 
	network.propagateBackwards(errorVector);


	network.propagateForwards(inputGrid);

	outputVector = network.getOutputs();
	for (vector<double>::iterator it = outputVector.begin(); it < outputVector.end(); it++)
	{
		cout << (*it) << " " << "\nEND";
	}

	return 0;
}

