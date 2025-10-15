#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <immintrin.h>

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

namespace
{

__m256i mandel_vec(__m256 c_re, __m256 c_im, int count)
{
    __m256 z_re = c_re, z_im = c_im;
    __m256i iter = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi32(1);
    const __m256 four = _mm256_set1_ps(4.f);
    const __m256 two = _mm256_set1_ps(2.f);

    for (int i = 0; i < count; ++i)
    {

        __m256 zr2 = _mm256_mul_ps(z_re, z_re);
        __m256 zi2 = _mm256_mul_ps(z_im, z_im);
        __m256 mag2 = _mm256_add_ps(zr2, zi2);

        // if (z_re * z_re + z_im * z_im > 4.f)
        //     break;
        __m256 active_ps = _mm256_cmp_ps(mag2, four, _CMP_LE_OQ);
        int mask = _mm256_movemask_ps(active_ps);
        if(mask == 0) break;

        __m256i active_i = _mm256_castps_si256(active_ps);
        iter = _mm256_add_epi32(iter, _mm256_and_si256(active_i, one)); // 0xFFFFFFFF = -1

        // float new_re = (z_re * z_re) - (z_im * z_im);
        __m256 new_re = _mm256_sub_ps(zr2, zi2);
        // float new_im = 2.f * z_re * z_im;
        __m256 new_im = _mm256_mul_ps(_mm256_mul_ps(two, z_re), z_im);
        // z_re = c_re + new_re;
        __m256 z_re_next = _mm256_add_ps(c_re, new_re);
        // z_im = c_im + new_im;
        __m256 z_im_next = _mm256_add_ps(c_im, new_im);

        z_re = _mm256_blendv_ps(z_re, z_re_next, active_ps);
        z_im = _mm256_blendv_ps(z_im, z_im_next, active_ps);

    }

    return iter;
}

}

void mandelbrot_serial(float x0,
                        float y0,
                        float x1,
                        float y1,
                        int width,
                        int height,
                        int start_row,
                        int num_rows,
                        int max_iterations,
                        int *output)
{
    float dx = (x1 - x0) / (float)width;
    float dy = (y1 - y0) / (float)height;

    int end_row = start_row + num_rows;

    for (int j = start_row; j < end_row; j++)
    {
        for (int i = 0; i < width; i+=8)
        {
            // float x = x0 + ((float)i * dx);
            __m256 x = _mm256_set_ps(
                x0+(i+7)*dx, x0+(i+6)*dx, x0+(i+5)*dx, x0+(i+4)*dx,
                x0+(i+3)*dx, x0+(i+2)*dx, x0+(i+1)*dx, x0+(i+0)*dx
            );
            // float y = y0 + ((float)j * dy);
            __m256 y = _mm256_set1_ps(y0+j*dy);

            // int index = ((j * width) + i);
            // output[index] = mandel(x, y, max_iterations);
            __m256i it = mandel_vec(x, y, max_iterations);
            
            alignas(32) int buf[8];
            _mm256_store_si256((__m256i*)buf, it);
            int row_base = j * width;
            for(int k = 0; k < 8; k++) {
                output[row_base + i + k] = buf[k];
            }

        }
    }
}

//
// worker_thread_start --
//
// Thread entrypoint.
void worker_thread_start(WorkerArgs *const args)
{

    // auto start = std::chrono::high_resolution_clock::now();

    // ======================= compute ===================================
    // Q1, Q2
    // int rows_per_thread = args->height / args->numThreads;
    // int start_row = args->threadId * rows_per_thread;
    // if(args->threadId == args->numThreads - 1) {
    //     rows_per_thread = args->height - start_row;
    // }
    // mandelbrot_serial(args->x0, args->y0, args->x1, args->y1, args->width, args->height, start_row, rows_per_thread, args->maxIterations, args->output);
    
    // Q3
    for (int row = args->threadId; row < (int)args->height; row += args->numThreads) {
        mandelbrot_serial(args->x0, args->y0, args->x1, args->y1, args->width, args->height, row, 1, args->maxIterations, args->output);
    }

    // ========================= output ====================================
    // Q1
    // printf("Thread %d finished rows [%d, %d)\n", args->threadId, start_row, start_row + rows_per_thread);
    
    // caculate time
    // auto end = std::chrono::high_resolution_clock::now();
    // double duration = std::chrono::duration<double, std::milli>(end - start).count();

    // Q2
    // printf("Thread %d finished rows [%d, %d) in %.3f ms\n", args->threadId, start_row, start_row + rows_per_thread, duration);

    // Q3
    // printf("Thread %d (interleaved) finished in %.3f ms\n", args->threadId, duration);
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
