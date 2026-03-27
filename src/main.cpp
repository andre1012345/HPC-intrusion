#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include "dataloader.h"
#include "knn.h"
#include "logistic_regression.h"
#include "random_forest.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. DATA LOADING (Always happens first)
    // We don't time this part because I/O is inherently serial
    if (rank == 0) std::cout << "Loading dataset..." << std::endl;
    std::vector<DataRow> data = loadCSV("data/cleaned_tuesday.csv");

    // 2. START THE HPC TIMER (Focus on Computation)
    MPI_Barrier(MPI_COMM_WORLD); 
    double startTime = MPI_Wtime();

    int v0 = 0, v1 = 0, v2 = 0; // Initialize to fix compiler warnings

    if (size == 1) {
        // --- SEQUENTIAL BASELINE ---
        // Increase epochs to 50 to make the CPU work harder
        v1 = predictLR(trainLogisticRegression(data, 50, 0.01), data[0].features);
        v2 = predictKNN(data, data[0].features, 3);
        v0 = predictRF(data, data[0].features);
    } 
    else {
        // --- HYBRID PARALLEL ---
        if (rank == 1) {
            v1 = predictLR(trainLogisticRegression(data, 50, 0.01), data[0].features);
            MPI_Send(&v1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
        else if (rank == 2) {
            v2 = predictKNN(data, data[0].features, 3);
            MPI_Send(&v2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
        else if (rank == 0) {
            v0 = predictRF(data, data[0].features);
            MPI_Recv(&v1, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&v2, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); 
    double endTime = MPI_Wtime();

    // 3. PERFORMANCE REPORT
    if (rank == 0) {
        double computeTime = endTime - startTime;
        int finalDecision = ((v0 + v1 + v2) >= 2) ? 1 : 0;
        
        std::cout << "\n=====================================" << std::endl;
        std::cout << "COMPUTATION METRICS (Ensemble)" << std::endl;
        std::cout << "Decision: " << (finalDecision ? "ATTACK 🚨" : "BENIGN ✅") << std::endl;
        std::cout << "Pure Compute Time: " << computeTime << " seconds" << std::endl;
        std::cout << "Speedup Proof: np=" << size << ", threads=" << omp_get_max_threads() << std::endl;
        std::cout << "=====================================\n" << std::endl;
    }

    MPI_Finalize();
    return 0;
}