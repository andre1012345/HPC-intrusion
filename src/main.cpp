#include <mpi.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // This is the Inter-Node part (MPI)
    printf("Hello from MPI Rank %d out of %d nodes\n", world_rank, world_size);

    // This is the Intra-Node part (OpenMP)
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        printf("  -> Rank %d: Thread %d out of %d threads\n", world_rank, thread_id, total_threads);
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}