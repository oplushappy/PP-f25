#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // the amount process need to do
    long long int local_tosses = tosses / world_size;
    if (world_rank == world_size - 1) {
        local_tosses += tosses % world_size;
    }

    // every process do
    unsigned int seed = (unsigned int)(time(NULL) ^ (world_rank * 1337u + 0x9e3779b9u));
    long long int local_count = 0;
    for (long long int i = 0; i < local_tosses; i++) {
        // rand_r thread safe
        double x = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 - 1.0;
        double d2 = x * x + y * y;
        if (d2 <= 1.0) local_count++;
    }
    // TODO: use MPI_Reduce
    // MPI_Reduce, MPI_Ireduce - Reduces values on all processes within a group.
    // int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
    long long int total_count = 0;
    MPI_Reduce(&local_count, &total_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);


    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4.0 * (double)total_count / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
