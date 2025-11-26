#include <cstdio>
#include <cuda.h>

#define STREAMS 8

__global__
void mandel_kernel(float lower_x, 
                float lower_y,
                float step_x, 
                float step_y,
                int res_x, 
                int res_y,
                int start_y, 
                int count_y,
                int * __restrict__ d_img,
                int max_iterations)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + start_y;

    if (x >= res_x || y >= start_y + count_y) return;

    float c_re = lower_x + x * step_x;
    float c_im = lower_y + y * step_y;

    float z_re = c_re, z_im = c_im;
    int i = 0;

    #pragma unroll 8
    for (; i < max_iterations; i++) {
        float re2 = z_re * z_re;
        float im2 = z_im * z_im;
        if (re2 + im2 > 4.f) break;

        float new_re = re2 - im2;
        float new_im = 2.f * z_re * z_im;

        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    d_img[y * res_x + x] = i;
}

void host_fe(float upper_x, 
            float upper_y,
            float lower_x, 
            float lower_y,
            int *img,
            int res_x, 
            int res_y,
            int max_iterations)
{
    // dx, dy
    float step_x = (upper_x - lower_x) / res_x;
    float step_y = (upper_y - lower_y) / res_y;

    size_t bytes = (size_t)res_x * res_y * sizeof(int);

    int *d_img;
    cudaMalloc(&d_img, bytes);

    cudaStream_t streams[STREAMS];
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // 1200 / 8 = 150
    int chunk = res_y / STREAMS;

    dim3 block(8, 8);
    // 1600 / 8 = 200, 150 / 8 = 19 , total 200 x 19 = 3800 blocks, 3800 block x 64 threads = 243,200 threads  
    dim3 grid((res_x + block.x - 1) / block.x, (chunk + block.y - 1) / block.y);

    int start_y = 0;

    for (int s = 0; s < STREAMS; s++) {
        int count_y = (s == STREAMS - 1) ? (res_y - start_y) : chunk;

        mandel_kernel<<<grid, block, 0, streams[s]>>>(lower_x, lower_y, step_x, step_y, res_x, res_y, start_y, count_y, d_img, max_iterations);

        start_y += chunk;
    }

    cudaMemcpy(img, d_img, bytes, cudaMemcpyDeviceToHost);

    for (int s = 0; s < STREAMS; s++) {
        cudaStreamDestroy(streams[s]);
    }

    cudaFree(d_img);
}
