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
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // the amount process need to do
    long long int total_count;
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

    // TODO: binary tree redunction
    for (int step = 1; step < world_size; step <<= 1) {
        if ((world_rank % (step << 1)) == 0) {
            // receieve right node（world_rank + step）
            long long int recv_count = 0;
            int src = world_rank + step;
            MPI_Recv(&recv_count, 1, MPI_LONG_LONG, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_count += recv_count;
        } else if ((world_rank % (step << 1)) == step) { // after first turn, node will not all need to send or receieve
            //  send left node（world_rank - step）
            int dst = world_rank - step;
            MPI_Send(&local_count, 1, MPI_LONG_LONG, dst, 0, MPI_COMM_WORLD);
            break;
        }
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        long long int total_count = local_count;
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
