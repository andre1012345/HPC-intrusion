#ifndef KNN_H
#define KNN_H

#include "dataloader.h"
#include <vector>

// k is the number of neighbors to check (usually 3 or 5)
int predictKNN(const std::vector<DataRow>& trainingData, const std::vector<double>& testFeatures, int k);

#endif