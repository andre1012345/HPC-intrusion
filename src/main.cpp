#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include "dataloader.h"
#include "knn.h"
#include "logistic_regression.h"
#include "random_forest.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. DATA LOADING (All ranks load for local training)
    std::vector<DataRow> data = loadCSV("data/cleaned_tuesday.csv");
    if (data.empty()) {
        if (rank == 0) std::cerr << "Error: CSV is empty or not found!" << std::endl;
        MPI_Finalize(); return 1;
    }

    // 2. WORKLOAD SETUP (1,000 packet batch for testing)
    int testSize = 100;
    int totalRows = data.size();
    if (totalRows < testSize) testSize = totalRows / 2; // Safety for small datasets
    int trainSize = totalRows - testSize;

    // Buffers for predictions
    std::vector<int> v0(testSize, 0), v1(testSize, 0), v2(testSize, 0);

    // --- START TIMER ---
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // =================================================================
    // SIMULATION MODE 1: SEQUENTIAL BATCH (1 Rank)
    // =================================================================
    if (size == 1) {
        std::cout << "[SEQUENTIAL] Processing " << testSize << " packets..." << std::endl;
        
        // Train and Predict LR
        Model lrModel = trainLogisticRegression(data, 50, 0.01);
        for(int i = 0; i < testSize; ++i) v1[i] = predictLR(lrModel, data[trainSize + i].features);

        // Predict KNN
        for(int i = 0; i < testSize; ++i) v2[i] = predictKNN(data, data[trainSize + i].features, 3);

        // Predict RF
        for(int i = 0; i < testSize; ++i) v0[i] = predictRF(data, data[trainSize + i].features);
    } 
    // =================================================================
    // SIMULATION MODE 2: HYBRID PARALLEL BATCH (3 Ranks)
    // =================================================================
    else if (size >= 3) {
        if (rank == 1) {
            Model lrModel = trainLogisticRegression(data, 50, 0.01);
            for(int i = 0; i < testSize; ++i) v1[i] = predictLR(lrModel, data[trainSize + i].features);
            MPI_Send(v1.data(), testSize, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
        else if (rank == 2) {
            for(int i = 0; i < testSize; ++i) v2[i] = predictKNN(data, data[trainSize + i].features, 3);
            MPI_Send(v2.data(), testSize, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
        else if (rank == 0) {
            // Rank 0 does RF locally
            for(int i = 0; i < testSize; ++i) v0[i] = predictRF(data, data[trainSize + i].features);
            
            // Catch results from workers
            MPI_Recv(v1.data(), testSize, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(v2.data(), testSize, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // --- STOP TIMER ---
    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();

    // 3. RESULTS & ACCURACY (Rank 0 only)
    if (rank == 0) {
        int correct = 0;
        for (int i = 0; i < testSize; ++i) {
            int ensembleVote = ((v0[i] + v1[i] + v2[i]) >= 2) ? 1 : 0;
            int actual = std::stoi(data[trainSize + i].label);
            if (ensembleVote == actual) correct++;
        }

        double duration = endTime - startTime;
        std::cout << "\n========== FINAL PERFORMANCE REPORT ==========" << std::endl;
        std::cout << "Ranks: " << size << " | Threads/Rank: " << omp_get_max_threads() << std::endl;
        std::cout << "Batch Size: " << testSize << " packets" << std::endl;
        std::cout << "Accuracy:   " << (double)correct / testSize * 100 << "%" << std::endl;
        std::cout << "Throughput: " << (double)(testSize / duration) << " packets/sec" << std::endl;
        std::cout << "Time:       " << duration << " seconds" << std::endl;
        std::cout << "==============================================\n" << std::endl;
    }

    MPI_Finalize();
    return 0;
}