#include "knn.h"
#include <cmath>
#include <algorithm>
#include <map>
#include <omp.h>

double getEuclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

int predictKNN(const std::vector<DataRow>& trainingData, const std::vector<double>& testFeatures, int k) {
    int n = trainingData.size();
    // Stores {distance, label}
    std::vector<std::pair<double, int>> distances(n);

    // --- HPC POWER: OPENMP PARALLELISM ---
    // This loop calculates distances for 445k+ rows in parallel
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double dist = getEuclideanDistance(trainingData[i].features, testFeatures);
        distances[i] = {dist, std::stoi(trainingData[i].label)};
    }

    // Sort to find the 'k' closest neighbors
    std::sort(distances.begin(), distances.end());

    // Majority vote
    std::map<int, int> counts;
    for (int i = 0; i < k; ++i) {
        counts[distances[i].second]++;
    }

    int finalLabel = -1;
    int maxVotes = -1;
    for (auto const& [label, count] : counts) {
        if (count > maxVotes) {
            maxVotes = count;
            finalLabel = label;
        }
    }

    return finalLabel;
}