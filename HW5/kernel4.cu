#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cstring>  

#ifndef GROUP_SIZE
#define GROUP_SIZE 4
#endif

__global__ void mandel_kernel(float lower_x, 
                            float lower_y,
                            float step_x, 
                            float step_y,
                            unsigned char *d_img,
                            size_t pitch,
                            int res_x, 
                            int res_y,
                            int max_iterations)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    
    int group_base_x = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_SIZE;
    // int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisY >= res_y) return;

    // Map pixel to complex plane
    // float x = x0 + ((float)i * dx); c_re
    // float y = y0 + ((float)j * dy); c_im
    float c_im = lower_y + thisY * step_y;

    int *row_ptr = (int*)(d_img + thisY * pitch);
    for(int k = 0; k < GROUP_SIZE; ++k) {
        int thisX = group_base_x + k;
        if (thisX >= res_x) break;

        float c_re = lower_x + thisX * step_x;

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

        row_ptr[thisX] = i;
    }
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
    // pitch
    const size_t row_bytes  = (size_t)res_x * sizeof(int);

    // (1) Host buffer: cudaHostAlloc
    // Allocates page-locked memory on the host.
    // __host__​cudaError_t cudaHostAlloc ( void** pHost, size_t size, unsigned int  flags )
    // cudaHostAllocDefault: This flag's value is defined to be 0 and causes cudaHostAlloc() to emulate cudaMallocHost().
    int *h_img = nullptr;
    cudaHostAlloc((void**)&h_img, bytes, cudaHostAllocDefault);

    // (2) Device buffer: cudaMallocPitch
    // __host__​cudaError_t cudaMallocPitch ( void** devPtr, size_t* pitch, size_t width, size_t height )
    // Allocates pitched memory on the device.
    // width - Requested pitched allocation width (in bytes)
    // height - Requested pitched allocation height
    unsigned char *d_img = nullptr;
    size_t pitch = 0;
    cudaMallocPitch((void**)&d_img, &pitch, row_bytes, res_y);
    // row0: [ data data ... data ][ padding ... ]
    // row1: [ data data ... data ][ padding ... ]
    // row2: [ data data ... data ][ padding ... ]
    // pitch = 每列「實際分配」大小（有效資料 + padding）

    // (3) Launch: 1 thread GROUP_SIZE pixel
    // ceil(a / b) = (a + b - 1) / b
    dim3 block(16, 16);
    int groups_x = (res_x + GROUP_SIZE - 1) / GROUP_SIZE;
    dim3 grid((groups_x + block.x - 1) / block.x, (res_y + block.y - 1) / block.y);

    mandel_kernel<<<grid, block>>>(lower_x, lower_y, step_x, step_y, d_img, pitch, res_x, res_y, max_iterations);
    cudaDeviceSynchronize();

    // (4) Copy back to host buffer
    // __host__​cudaError_t cudaMemcpy2D ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind )
    // dpitch - Pitch of destination memory
    // spitch - Pitch of source memory
    cudaMemcpy2D(h_img, row_bytes, d_img, pitch, row_bytes, res_y, cudaMemcpyDeviceToHost);

    // (5) Copy to output img (provided by main)
    // void * memcpy ( void * destination, const void * source, size_t num ); num bytes
    std::memcpy(img, h_img, bytes);

    // (6) Free
    // __host__​__device__​cudaError_t cudaFree ( void* devPtr )
    // Frees memory on the device.
    cudaFree(d_img);
    cudaFreeHost(h_img);

}
