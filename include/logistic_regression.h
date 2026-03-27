#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "dataloader.h"
#include <vector>

struct Model {
    std::vector<double> weights;
    double bias;
};

// Simple Gradient Descent training
Model trainLogisticRegression(const std::vector<DataRow>& data, int epochs, double learningRate);

// Predict 0 or 1
int predictLR(const Model& model, const std::vector<double>& features);

#endif