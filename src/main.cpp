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

    // --- T0: TOTAL START ---
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_start = MPI_Wtime();

    // --- 7.1 & 7.2: DATA LOADING PHASE ---
    double t_load_start = MPI_Wtime();
    
    // Every rank loads data for local processing (Simple strategy)
    // To monitor the I/O bottleneck, we time this specifically
    std::vector<DataRow> data = loadCSV("data/cleaned_tuesday.csv");
    
    double t_load_end = MPI_Wtime();
    double loadTime = t_load_end - t_load_start;

    // --- 7.4: COMPUTATION PHASE ---
    MPI_Barrier(MPI_COMM_WORLD); // Sync before math starts
    double t_comp_start = MPI_Wtime();

    int v0 = 0, v1 = 0, v2 = 0;

    if (size == 1) {
        // Sequential Baseline
        v1 = predictLR(trainLogisticRegression(data, 50, 0.01), data[0].features);
        v2 = predictKNN(data, data[0].features, 3);
        v0 = predictRF(data, data[0].features);
    } 
    else {
        // Hybrid Parallel
        if (rank == 1) {
            v1 = predictLR(trainLogisticRegression(data, 50, 0.01), data[0].features);
            MPI_Send(&v1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        } else if (rank == 2) {
            v2 = predictKNN(data, data[0].features, 3);
            MPI_Send(&v2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        } else if (rank == 0) {
            v0 = predictRF(data, data[0].features);
            MPI_Recv(&v1, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&v2, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    double t_comp_end = MPI_Wtime();
    double computeTime = t_comp_end - t_comp_start;

    // --- T_FINAL: TOTAL END ---
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_end = MPI_Wtime();
    double totalExecutionTime = t_total_end - t_total_start;

    // --- 7.4: PERFORMANCE EVALUATION REPORT ---
    if (rank == 0) {
        std::cout << "\n========== HPC PERFORMANCE REPORT ==========" << std::endl;
        std::cout << "Configuration: " << size << " Ranks | " << omp_get_max_threads() << " Threads" << std::endl;
        std::cout << "--------------------------------------------" << std::endl;
        std::cout << "1. DATA LOADING TIME:    " << loadTime << " s" << std::endl;
        std::cout << "2. PURE COMPUTE TIME:    " << computeTime << " s" << std::endl;
        std::cout << "3. TOTAL PIPELINE TIME:  " << totalExecutionTime << " s" << std::endl;
        std::cout << "--------------------------------------------" << std::endl;
        
        // Comm-to-Comp Ratio calculation
        double commTime = totalExecutionTime - computeTime - loadTime;
        if (commTime < 0) commTime = 0; // Floating point precision safety
        std::cout << "4. COMM-TO-COMP RATIO:   " << (commTime / computeTime) << std::endl;
        
        std::cout << "FINAL DECISION: " << (((v0+v1+v2)>=2) ? "ATTACK 🚨" : "BENIGN ✅") << std::endl;
        std::cout << "============================================\n" << std::endl;
    }

    MPI_Finalize();
    return 0;
}