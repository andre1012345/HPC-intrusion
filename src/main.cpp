#include <mpi.h>
#include <omp.h>
#include <iostream>
#include "dataloader.h"
#include "knn.h"
#include "logistic_regression.h"

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
        std::cout << "[RANK 1] Logistic Regression Engine loading data..." << std::endl;
        std::vector<DataRow> lrData = loadCSV("data/cleaned_tuesday.csv");
        
        if (!lrData.empty()) {
            std::cout << "[RANK 1] Training Model (Gradient Descent)..." << std::endl;
            Model myModel = trainLogisticRegression(lrData, 10, 0.01);
            
            int pred = predictLR(myModel, lrData[0].features);
            std::cout << "[RANK 1] Prediction: " << pred << std::endl;
        }
    }
  else if (rank == 2) {
        std::cout << "[RANK 2] K-NN Engine active. Loading data..." << std::endl;
        std::vector<DataRow> trainData = loadCSV("data/cleaned_tuesday.csv");

        if(!trainData.empty()) {
            // Let's test K-NN with the first row as a 'new' packet
            std::vector<double> testPacket = trainData[0].features;
            int actualLabel = std::stoi(trainData[0].label);

            std::cout << "[RANK 2] Classifying test packet..." << std::endl;
            
            // Run K-NN with k=3
            int prediction = predictKNN(trainData, testPacket, 3);

            std::cout << "[RANK 2] Prediction: " << prediction 
                      << " | Actual: " << actualLabel << std::endl;
        }
    }

    // 4. Close MPI
    MPI_Finalize();
    return 0;
}