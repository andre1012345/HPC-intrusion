#include <mpi.h>
#include <omp.h>
#include <iostream>
#include "dataloader.h"

int main(int argc, char** argv) {
    // 1. Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 3) {
        if (rank == 0) std::cout << "This project requires 3 ranks (0: RF, 1: LR, 2: KNN)" << std::endl;
        MPI_Finalize();
        return 0;
    }

    // 2. Load the Cleaned Data
    // In a real HPC scenario, Rank 0 might load and scatter, 
    // but for now, each rank loads its own copy for simplicity.
    std::string path = "data/cleaned_tuesday.csv";
    std::vector<DataRow> myData = loadCSV(path);

    if (myData.empty()) {
        std::cout << "Rank " << rank << " failed to load data." << std::endl;
        MPI_Finalize();
        return 0;
    }

    // 3. Assign Tasks
    if (rank == 0) {
        std::cout << "[RANK 0] starting Random Forest training on " << myData.size() << " rows..." << std::endl;
        // Call your Random Forest function here
    } 
    else if (rank == 1) {
        std::cout << "[RANK 1] starting Logistic Regression training..." << std::endl;
        // Call your Logistic Regression function here
    }
    else if (rank == 2) {
        std::cout << "[RANK 2] starting K-Nearest Neighbors training..." << std::endl;
        // Call your KNN function here
    }

    // 4. Close MPI
    MPI_Finalize();
    return 0;
}