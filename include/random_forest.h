#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "dataloader.h"
#include <vector>

struct Node {
    bool isLeaf;
    int featureIndex;
    double threshold;
    int prediction;
};

// Simplified Decision Tree for the Forest
class DecisionTree {
public:
    Node root;
    void train(const std::vector<DataRow>& data);
    int predict(const std::vector<double>& features) const;
};

// The Random Forest aggregator
int predictRF(const std::vector<DataRow>& data, const std::vector<double>& testPacket);

#endif