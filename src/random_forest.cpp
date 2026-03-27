#include "random_forest.h"
#include <algorithm>
#include <omp.h>
#include <map>

// Simplified Gini Calculation (The "Intensive Loop")
double calculateGini(const std::vector<DataRow>& data) {
    if (data.empty()) return 0.0;
    std::map<int, int> counts;
    for (const auto& row : data) counts[std::stoi(row.label)]++;
    
    double score = 0.0;
    for (auto const& [label, count] : counts) {
        double p = (double)count / data.size();
        score += p * p;
    }
    return 1.0 - score;
}

int predictRF(const std::vector<DataRow>& data, const std::vector<double>& testPacket) {
    // In a real Random Forest, we'd build multiple trees. 
    // Here, we parallelize the search for the best split across all features.
    int bestFeature = 0;
    double bestGini = 1.0;
    int numFeatures = data[0].features.size();

    // HPC MOMENT: Parallelize the search for the best feature split 
    #pragma omp parallel for
    for (int i = 0; i < numFeatures; ++i) {
        // (Simplified logic: finding the feature that minimizes Gini impurity)
        double currentGini = calculateGini(data); 
        #pragma omp critical
        if (currentGini < bestGini) {
            bestGini = currentGini;
            bestFeature = i;
        }
    }

    // Return prediction based on the best feature of the root node
    return (testPacket[bestFeature] > 0.5) ? 1 : 0;
}