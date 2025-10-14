#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>

struct WorkerArgs
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
};

extern void mandelbrot_serial(float x0,
                              float y0,
                              float x1,
                              float y1,
                              int width,
                              int height,
                              int start_row,
                              int num_rows,
                              int max_iterations,
                              int *output);

//
// worker_thread_start --
//
// Thread entrypoint.
void worker_thread_start(WorkerArgs *const args)
{

    auto start = std::chrono::high_resolution_clock::now();

    // ======================= compute ===================================
    // Q1, Q2
    // int rows_per_thread = args->height / args->numThreads;
    // int start_row = args->threadId * rows_per_thread;
    // if(args->threadId == args->numThreads - 1) {
    //     rows_per_thread = args->height - start_row;
    // }
    // mandelbrot_serial(args->x0, args->y0, args->x1, args->y1, args->width, args->height, start_row, rows_per_thread, args->maxIterations, args->output);
    
    // Q3
    for (int row = args->threadId; row < args->height; row += args->numThreads) {
        mandelbrot_serial(args->x0, args->y0, args->x1, args->y1, args->width, args->height, row, 1, args->maxIterations, args->output);
    }

    // ========================= output ====================================
    // Q1
    // printf("Thread %d finished rows [%d, %d)\n", args->threadId, start_row, start_row + rows_per_thread);
    
    // caculate time
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();

    // Q2
    // printf("Thread %d finished rows [%d, %d) in %.3f ms\n", args->threadId, start_row, start_row + rows_per_thread, duration);

    // Q3
    printf("Thread %d (interleaved) finished in %.3f ms\n", args->threadId, duration);
}

//
// mandelbrot_thread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrot_thread(int num_threads,
                       float x0,
                       float y0,
                       float x1,
                       float y1,
                       int width,
                       int height,
                       int max_iterations,
                       int *output)
{
    static constexpr int max_threads = 32;

    if (num_threads > max_threads)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", max_threads);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::array<std::thread, max_threads> workers;
    std::array<WorkerArgs, max_threads> args = {};

    for (int i = 0; i < num_threads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = max_iterations;
        args[i].numThreads = num_threads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < num_threads; i++)
    {
        workers[i] = std::thread(worker_thread_start, &args[i]);
    }

    worker_thread_start(&args[0]);

    // join worker threads
    for (int i = 1; i < num_threads; i++)
    {
        workers[i].join();
    }
}
