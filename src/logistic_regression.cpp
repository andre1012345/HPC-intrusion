#include "logistic_regression.h"
#include <cmath>
#include <omp.h>

double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

Model trainLogisticRegression(const std::vector<DataRow>& data, int epochs, double learningRate) {
    int numFeatures = data[0].features.size();
    Model model;
    model.weights.resize(numFeatures, 0.0);
    model.bias = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double dW_sum = 0.0; // Simplified for this example
        double dB_sum = 0.0;

        // HPC MOMENT: Parallelize the gradient calculation
        #pragma omp parallel for reduction(+:dB_sum)
        for (size_t i = 0; i < data.size(); ++i) {
            double z = model.bias;
            for (int j = 0; j < numFeatures; ++j) {
                z += model.weights[j] * data[i].features[j];
            }
            double prediction = sigmoid(z);
            double error = prediction - std::stoi(data[i].label);
            
            dB_sum += error;
            // In a full version, we'd update every weight here in a parallel-safe way
        }

        model.bias -= learningRate * (dB_sum / data.size());
    }
    return model;
}

int predictLR(const Model& model, const std::vector<double>& features) {
    double z = model.bias;
    for (size_t i = 0; i < features.size(); ++i) {
        z += model.weights[i] * features[i];
    }
    return (sigmoid(z) >= 0.5) ? 1 : 0;
}