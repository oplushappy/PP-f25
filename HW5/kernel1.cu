#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cstring>  

__global__ void mandel_kernel(float lower_x, 
                            float lower_y,
                            float step_x, 
                            float step_y,
                            int *d_img,
                            int res_x, 
                            int res_y,
                            int max_iterations)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisX >= res_x || thisY >= res_y) return;

    // Map pixel to complex plane
    // float x = x0 + ((float)i * dx); c_re
    // float y = y0 + ((float)j * dy); c_im
    float c_re = lower_x + thisX * step_x;
    float c_im = lower_y + thisY * step_y;

    // Inline mandel() computation (same as serial)
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < max_iterations; ++i)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = (z_re * z_re) - (z_im * z_im);
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    // int index = ((j * width) + i);
    int idx = thisY * res_x + thisX;
    d_img[idx] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
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
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    const size_t num_pixels = (size_t)res_x * (size_t)res_y;
    const size_t bytes = num_pixels * sizeof(int);

    // (1) Host buffer: MUST use new, NOT img directly
    int *h_img = new int[num_pixels];

    // (2) Device buffer: cudaMalloc
    // __device__​cudaError_t cudaMalloc ( void** devPtr, size_t size )
    int *d_img = nullptr;
    cudaMalloc((void**)&d_img, bytes);

    // (3) Launch: 1 thread per pixel
    // ceil(a / b) = (a + b - 1) / b
    dim3 block(16, 16);
    dim3 grid((res_x + block.x - 1) / block.x, (res_y + block.y - 1) / block.y);

    mandel_kernel<<<grid, block>>>(lower_x, lower_y, step_x, step_y, d_img, res_x, res_y, max_iterations);
    cudaDeviceSynchronize();

    // (4) Copy back to host buffer
    // __host__​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
    // Copies data between host and device.
    cudaMemcpy(h_img, d_img, bytes, cudaMemcpyDeviceToHost);

    // (5) Copy to output img (provided by main)
    // void * memcpy ( void * destination, const void * source, size_t num ); num bytes
    std::memcpy(img, h_img, bytes);

    // (6) Free
    // __host__​__device__​cudaError_t cudaFree ( void* devPtr )
    // Frees memory on the device.
    cudaFree(d_img);
    delete[] h_img;

}
