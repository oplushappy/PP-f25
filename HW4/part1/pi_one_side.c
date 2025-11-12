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

    MPI_Win win;

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

    // https://enccs.github.io/intermediate-mpi/one-sided-concepts/
    // processes can access data on other processes, as long as it is made available in special memory windows.
    // MPI_Win_create - One-sided MPI call that returns a window object for RMA operations.
    // MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win)
    // int MPI_Win_create(
    //     void *base,          // 被公開的記憶體位址
    //     MPI_Aint size,       // 記憶體大小（byte）
    //     int disp_unit,       // 單位距離（通常是 sizeof(element)）
    //     MPI_Info info,       // 可給 MPI 的提示（通常填 MPI_INFO_NULL）
    //     MPI_Comm comm,       // 通訊群組
    //     MPI_Win *win         // 輸出參數：window 物件
    // );
    long long int total_count = 0;
    if (world_rank == 0)
    {
        // Main
        MPI_Win_create(&total_count, sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    else
    {
        // Workers
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    // 所有 rank 對 root(0) 做 one-sided 原子加總
    // MPI_Win_lock - Starts an RMA access epoch locking access to a particular rank.
    // int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win)
    // lock_type Either MPI_LOCK_EXCLUSIVE or MPI_LOCK_SHARED (state).
    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
    // MPI_Accumulate, MPI_Raccumulate - Combines the contents of the origin buffer with that of a target buffer.
    // int MPI_Accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank,
    // MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)
    // target_disp 目標位移（offset）。若目標 window 是單一變數，就填 0
    MPI_Accumulate(&local_count, 1, MPI_LONG_LONG, 0, 0, 1, MPI_LONG_LONG, MPI_SUM, win);
    // int MPI_Win_unlock(int rank, MPI_Win win)
    MPI_Win_unlock(0, win);

    // 同步：確保所有 accumulate 都完成
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
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
