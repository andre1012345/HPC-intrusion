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

    if (size < 3) {
        if (rank == 0) std::cout << "Error: This project requires at least 3 ranks." << std::endl;
        MPI_Finalize();
        return 0;
    }

    // --- RANK 1: Logistic Regression ---
    if (rank == 1) {
        std::vector<DataRow> lrData = loadCSV("data/cleaned_tuesday.csv");
        Model myModel = trainLogisticRegression(lrData, 5, 0.01);
        int prediction = predictLR(myModel, lrData[0].features);

        // SEND prediction to Rank 0
        MPI_Send(&prediction, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        std::cout << "[RANK 1] Sent prediction (" << prediction << ") to Rank 0" << std::endl;
    }

    // --- RANK 2: K-Nearest Neighbors ---
    else if (rank == 2) {
        std::vector<DataRow> knnData = loadCSV("data/cleaned_tuesday.csv");
        int prediction = predictKNN(knnData, knnData[0].features, 3);

        // SEND prediction to Rank 0
        MPI_Send(&prediction, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        std::cout << "[RANK 2] Sent prediction (" << prediction << ") to Rank 0" << std::endl;
    }

    // --- RANK 0: Random Forest Worker & Aggregator ---
    if (rank == 0) {
        // A. WORKER TASK: Train and predict using Random Forest [cite: 47]
        std::cout << "[RANK 0] Executing Random Forest training..." << std::endl;
        std::vector<DataRow> rfData = loadCSV("data/cleaned_tuesday.csv");
        
        int rfVote = 0;
        if (!rfData.empty()) {
            // Predict on the first packet as a test
            rfVote = predictRF(rfData, rfData[0].features);
            std::cout << "[RANK 0] RF Local Prediction: " << rfVote << std::endl;
        }

        // B. AGGREGATOR TASK: Collect votes via MPI_Recv [cite: 48, 56]
        int vote1, vote2;
        std::cout << "[RANK 0] Waiting for Rank 1 (LR) and Rank 2 (KNN)..." << std::endl;
        
        // Receive from Rank 1 (Logistic Regression) [cite: 56]
        MPI_Recv(&vote1, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Receive from Rank 2 (K-NN) [cite: 56]
        MPI_Recv(&vote2, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // C. ENSEMBLE DECISION: Majority Vote (2 out of 3) [cite: 9, 57]
        int totalAttackVotes = rfVote + vote1 + vote2;

        std::cout << "\n=====================================" << std::endl;
        std::cout << "   HETEROGENEOUS ENSEMBLE REPORT     " << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "RF Vote (Rank 0): " << rfVote << std::endl;
        std::cout << "LR Vote (Rank 1): " << vote1 << std::endl;
        std::cout << "KNN Vote (Rank 2): " << vote2 << std::endl;
        std::cout << "-------------------------------------" << std::endl;

        if (totalAttackVotes >= 2) {
            std::cout << "FINAL VERDICT: ATTACK DETECTED! 🚨" << std::endl;
        } else {
            std::cout << "FINAL VERDICT: TRAFFIC IS BENIGN. ✅" << std::endl;
        }
        std::cout << "=====================================\n" << std::endl;
    }

    MPI_Finalize();
    return 0;
}