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

    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Send(&local_count, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.

        // MPI_Request requests[];
        int nrecv = world_size - 1;
        long long int *recvbuf = (long long int*)malloc(sizeof(long long int) * nrecv);
        // 未完成的非阻塞通訊的憑證
        MPI_Request *reqs = (MPI_Request*)malloc(sizeof(MPI_Request) * nrecv);

        for (int src = 1; src < world_size; ++src) {
            // MPI_Irecv - Starts a standard-mode, nonblocking receive
            // int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
            // will direct return , and suspend a request
            MPI_Irecv(&recvbuf[src - 1], 1, MPI_LONG_LONG, src, 0, MPI_COMM_WORLD, &reqs[src - 1]);
        }

        // MPI_Waitall();
        MPI_Waitall(nrecv, reqs, MPI_STATUSES_IGNORE);

        for (int i = 0; i < nrecv; ++i){
            local_count += recvbuf[i];
        }

        free(reqs);
        free(recvbuf);
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
