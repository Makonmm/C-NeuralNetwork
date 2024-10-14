#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include <fstream>

using namespace std;

// consts

#define LR 0.001
#define PRINT_PADDING 4
#define EPOCHS 1500
#define EPOCHS_PRINTS 10

class Layer;

class Node {
    public:
        int idInLayer;
        double value;
        double delta;
        vector<double> leftWeights;
        vector<double> gradients;


        Node(int leftUnits, int idInLayer) {
            this -> idInLayer = idInLayer;

            for (int i = 0; i < leftUnits; i++) 
                leftWeights.push_back(rand()/(double)RAND_MAX);

            gradients.resize(leftUnits, 0);
        }

        void updateWeights() {
            for (int i = 0; i < leftWeights.size(); i++)
                leftWeights[i] -= LR * gradients[i];

            fill(gradients.begin(), gradients.end(), 0);
        }

        void printWeights() {
            for (int i = 0; i < leftWeights.size(); i++)
                cout << "NODE: " << idInLayer << " " << "WEIGHT "<< i << ": " << leftWeights[i] << endl;
    
        }

        void predict(Layer leftLayer);
        void predictOutput(Layer leftLayer);
        void calculateGradients(Layer leftLayer, double loss);
        void calculateGradients(Layer leftLayer, Layer rightLayer);

};


vector<double> softmax(vector<double> values) {
    vector<double> result;

    double denominator = 0;

    for (int i = 0; i < values.size(); i++) 
        denominator += exp(values[i]);

    for (int i = 0; i < values.size(); i++)
        result.push_back(exp(values[i]) / denominator);

    return result;
}

double sigmoid(double innerProduct) {
    return 1 / (1 + exp(-innerProduct));
}

double sigmoidDerivative(double sigmoid) {
    return sigmoid * (1 - sigmoid);
}

double lossDerivativeWRTOutput(vector<double> outputValues, vector<int> labels, int outputIndex) {
    double sum = 0;

        // CROSS ENTROPY LOSS

    for (int i = 0; i < outputValues.size(); i++) {
        if (i == outputIndex) {
            sum += labels[i] * (outputValues[outputIndex] - 1);
        } else {
            sum += labels[i] * outputValues[outputIndex];
        }

    }

    return sum;
}

double innerProduct(vector<Node> v1, vector<double> v2) {
    double sum = 0;

    for (int i = 0; i < v1.size(); i++)
        sum+= v1[i].value * v2[i];

    return sum;
}

class Layer {
    public:
        int layerId;
        bool outputLayer;
        vector<Node> nodes;
        vector<double> outputValues;

        Layer(int units, int unitsLeft, bool bias, int layerId) {
            this -> layerId = layerId;
            this -> outputLayer = !bias;

            for (int i = 0; i < units; i++) {
                Node *node = new Node(unitsLeft + 1, i);
                nodes.push_back(*node);
            }

            if (bias) {
                Node *node = new Node(0, units);
                node -> value = 1;
                nodes.push_back(*node);
            }
        }

        void predict(Layer layer) {
            if(!outputLayer) {
                for(int i = 0; i < nodes.size(); i++)
                    nodes[i].predict(layer);
            } else {
                vector<double> outputValues;

                for (int i = 0; i < nodes.size(); i++) {
                    nodes[i].predictOutput(layer);
                    outputValues.push_back(nodes[i].value);

                }
                outputValues = softmax(outputValues);

                this -> outputValues = outputValues;

                for (int i = 0; i < nodes.size(); i++) 
                    nodes[i].value = outputValues[i];

            }
        }

        void calculateGradients(Layer leftLayer, vector<int> labels) {
            for (int i = 0; i < nodes.size(); i++)
                nodes[i].calculateGradients(leftLayer, lossDerivativeWRTOutput(this -> outputValues, labels, i));
        }

        void calculateGradients(Layer leftLayer, Layer rightLayer) {
            for (int i = 0; i < nodes.size(); i++) 
                nodes[i].calculateGradients(leftLayer, rightLayer);
        }

        void updateWeights() {
            for (int i = 0; i < nodes.size(); i++) 
                nodes[i].updateWeights();
        }

        void printWeights() {
            cout << "LAYER ID = " << layerId << endl;

            for (int i = 0; i < nodes.size(); i++)
                nodes[i].printWeights();
        }

};


void Node::calculateGradients(Layer leftLayer, Layer rightLayer) {

    if (leftWeights.size() > 0) {
        delta = sigmoidDerivative(value);

        double nodeError = 0;

        for (int i = 0; i < rightLayer.nodes.size(); i++) {
            if (rightLayer.nodes[rightLayer.nodes.size()-1].leftWeights.size() > 0)
                nodeError += rightLayer.nodes[i].delta * rightLayer.nodes[i].leftWeights[idInLayer];
        }

        delta *= nodeError;

        for (int i = 0; i < gradients.size(); i++)
            gradients[i] += leftLayer.nodes[i].value * delta;
    }
}

void Node::calculateGradients(Layer leftLayer, double loss) {
    delta = loss;

    for(int i = 0; i < leftWeights.size(); i++) 
        gradients[i] += leftLayer.nodes[i].value * delta;
}

void Node::predict(Layer leftLayer) {
    if (leftWeights.size() > 0)
        value = sigmoid(innerProduct(leftLayer.nodes, leftWeights));
}

void Node::predictOutput(Layer leftLayer) {
    value = innerProduct(leftLayer.nodes, leftWeights);
}

class Network {

    public: 
        vector<Layer> layers;

        Network(vector<int> layerSizes) {
            Layer *l0 = new Layer(layerSizes[0], -1, true, 0);
            layers.push_back(*l0);


            for (int i = 0; i < layerSizes.size()-1; i++) {
                Layer *l;

                if (i + 1 == layerSizes.size() - 1) {
                    l = new Layer(layerSizes[i + 1], layerSizes[i], false, i + 1);
                }
                else {
                    l = new Layer(layerSizes[i + 1], layerSizes[i], true, i + 1);
                }

                layers.push_back(*l);
            }
        }

        vector<vector<double>> predict(vector<vector<double>> m) {
            int outputUnits = layers[layers.size()-1].nodes.size();

            vector<vector<double>> result;

            result.resize(m.size(), vector<double>(outputUnits));

            for (int i = 0; i < m.size(); i++) {
                predictInstance(m, i);

                for (int j = 0; j < outputUnits; j++) 
                    result[i][j] = layers[layers.size()-1].nodes[j].value;
            }

            return result;
        }

        void predictInstance(vector<vector<double>> m, int row) {

            for (int i = 0; i < m[0].size(); i++) {
                layers[0].nodes[i].value = m[row][i];
            }

            for (int i = 1; i < layers.size(); i++) {
                layers[i].predict(layers[i-1]);
            }
        }

        void fit(vector<vector<double>> m, vector<vector<int>> labels) {
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                if (epoch % ((EPOCHS + EPOCHS_PRINTS - 1) / EPOCHS_PRINTS) == 0) {
                    cout << "EPOCH = " << setfill(' ') << setw(PRINT_PADDING) << epoch << " || "; loss(m, labels);
                }

                for (int i = 0; i < m.size(); i++) {
                    predictInstance(m, i);

                    layers[layers.size()-1].calculateGradients(layers[layers.size()-2], labels[i]);

                    for (int j = layers.size()-2; j > 0; j--) {
                        layers[j].calculateGradients(layers[j-1], layers[j+1]);
                    }
                }

                for (int i = 0; i < layers.size(); i++) {
                    layers[i].updateWeights();
                }
            }

            cout << "EPOCH = " << setfill(' ') << setw(PRINT_PADDING) << EPOCHS << " || "; loss(m, labels);   
            cout << "WEIGHTS VALUES = " << endl;

            for (int i = 0; i < layers.size(); i++) {
                layers[i].printWeights();
            }     
        }

        void loss(vector<vector<double>> m, vector<vector<int>> labels) {

            vector<vector<double>> result = predict(m);

            double loss = 0, rowMax;
            float accuracy = 0;
            int labelIndex, rowMaxIndex;

            for (int i = 0; i < labels.size(); i++) {
                rowMax = 0;
                rowMaxIndex = -1;
                labelIndex = -1;

                for (int j = 0; j < labels[0].size(); j++) {

                    loss -= labels[i][j] * log2(result[i][j]);

                    if (labelIndex == rowMaxIndex) accuracy++;
                }
            }

            loss /= labels.size();
            accuracy /= labels.size();

            cout << "ERROR = " << loss << " ACCURACY --> " <<  accuracy << endl;
        }
};

int main() {

    // Define the size of each layer in the network: 
    // 4 neurons in the input layer, 4 in the hidden layer, and 3 in the output layer.
    vector<int> layerSizes = {4, 4, 3};
    Network net(layerSizes);

    string line;

    // Open the dataset file
    ifstream infile("data/dataset.txt");
    vector<vector<double>> m;   // Input data matrix
    vector<vector<int>> labels; // Labels matrix (one-hot encoded)

    // Read the dataset line by line
    while (getline(infile, line)) {
        stringstream ss(line);
        string in_line;
        vector<double> row;

        // Initialize the label as a zero vector of size 3
        vector<int> label(3, 0);

        // Split the line by commas, convert to double and store the input features
        while (getline(ss, in_line, ',')) {
            row.push_back(stod(in_line, 0));
        }

        // Use the last value in the row as the class label (0, 1, or 2)
        // Create a one-hot encoded label based on the class
        label[row.back()] = 1;

        // Store the label
        labels.push_back(label);

        // Remove the last element from the row (the label)
        row.pop_back();
        
        // Store the input row
        m.push_back(row);
    }

    // Train the neural network with the input data and labels
    cout << "Learning CURVE: " << endl;
    net.fit(m, labels);

    // Make predictions using the same input data
    vector<vector<double>> out = net.predict(m);

    // Display the model's predictions
    cout << "Model predictions: " << endl;

    for (int i = 0; i < out.size(); i++) {
        // Print the actual label in one-hot format
        cout << "[" << labels[i][0] << ", " << labels[i][1] << ", " << labels[i][2] << "] --> ";
        // Print the predicted output from the network
        cout << "[" << out[i][0] << ", " << out[i][1] << ", " << out[i][2] << "]";
        cout << endl;
    }

    return 0;
}
