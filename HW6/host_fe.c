#include "host_fe.h"
#include "helper.h"
#include <stdio.h>
#include <stdlib.h>

void host_fe(int filter_width,
             float *filter,
             int image_height,
             int image_width,
             float *input_image,
             float *output_image,
             cl_device_id *device,
             cl_context *context,
             cl_program *program)
{
    cl_int status;

    // 計算資料大小
    size_t image_size_bytes  = (size_t)image_height * (size_t)image_width * sizeof(float);
    size_t filter_size_bytes = (size_t)filter_width * (size_t)filter_width * sizeof(float);

    // 1. Command Queue
    // cl_command_queue clCreateCommandQueue( cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int* errcode_ret);
    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &status);

    // 2. Device Buffers
    // cl_mem clCreateBuffer( cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret);
    cl_mem d_input = clCreateBuffer(*context, CL_MEM_READ_ONLY, image_size_bytes, NULL, &status);
    cl_mem d_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, image_size_bytes, NULL, &status);
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, filter_size_bytes, NULL, &status);

    // 3. 寫入資料
    // cl_int clEnqueueWriteBuffer( cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
    status = clEnqueueWriteBuffer(queue, d_input, CL_FALSE, 0, image_size_bytes, input_image, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, d_filter, CL_FALSE, 0, filter_size_bytes, filter, 0, NULL, NULL);

    // 4. Create Kernel
    // cl_kernel clCreateKernel( cl_program program, const char* kernel_name, cl_int* errcode_ret);
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    // dynamic
    // 定義 Tile Size (必須與 Kernel 內的 TILE_SIZE 宏一致)
    int tile_size = 16;

    // 計算 Local Memory 需要的寬度: Tile寬度 + Filter寬度 - 1 (Halo)
    int buffer_w = tile_size + filter_width - 1;
    
    // 計算總 Bytes 數
    size_t local_mem_size = buffer_w * buffer_w * sizeof(float);

    // 設定參數
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_filter);
    status |= clSetKernelArg(kernel, 3, sizeof(int),    &image_width);
    status |= clSetKernelArg(kernel, 4, sizeof(int),    &image_height);
    status |= clSetKernelArg(kernel, 5, sizeof(int),    &filter_width);
    
    // 告訴 OpenCL 分配 local_mem_size 大小的空間
    // 最後一個參數傳 NULL 代表這是動態分配的 Local Memory
    status |= clSetKernelArg(kernel, 6, local_mem_size, NULL); 
    
    // =========================================================================

    // 5. 設定 Work Group 大小
    size_t local_work_size[2] = {16, 16};
    size_t global_work_size[2];

    // Padding: Global Size 必須是 Local Size 的倍數
    // 公式： (size + local_size - 1) / local_size * local_size
    global_work_size[0] = (image_width + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0];
    global_work_size[1] = (image_height + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1];

    // 6. 執行 Kernel
    // cl_int clEnqueueNDRangeKernel( cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    // 7. 讀回結果
    // cl_int clEnqueueReadBuffer( cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
    status = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, image_size_bytes, output_image, 0, NULL, NULL);

    // 8. 釋放資源
    clReleaseKernel(kernel);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_filter);
    clReleaseCommandQueue(queue);
}