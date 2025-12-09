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

    size_t image_size  = (size_t)image_height * (size_t)image_width * sizeof(float);
    size_t filter_size = (size_t)filter_width * (size_t)filter_width * sizeof(float);

    // 1. 建立 command queue
    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &status);
    // CHECK(status, "clCreateCommandQueue");

    // 2. 建立 device buffer
    cl_mem d_input = clCreateBuffer(*context, CL_MEM_READ_ONLY,  image_size,  NULL, &status);
    cl_mem d_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, image_size, NULL, &status);
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY,  filter_size, NULL, &status);

    // 3. 將 host 資料寫入 device
    status = clEnqueueWriteBuffer(queue, d_input,  CL_FALSE, 0, image_size,  input_image, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, d_filter, CL_FALSE, 0, filter_size, filter,      0, NULL, NULL);

    // 4. 建立 kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    // 5. 設定 kernel 參數 (0~5)
    int w  = image_width;
    int h  = image_height;
    int fw = filter_width;

    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_filter);
    status |= clSetKernelArg(kernel, 3, sizeof(int),    &w);
    status |= clSetKernelArg(kernel, 4, sizeof(int),    &h);
    status |= clSetKernelArg(kernel, 5, sizeof(int),    &fw);
    // CHECK(status, "clSetKernelArg");

    // 6. 設定 local / global work size（tile 大小）
    const size_t local_work_size[2] = { 16, 16 };   // 先宣告再用

    size_t global_work_size[2];
    global_work_size[0] =
        ((size_t)image_width  + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0];
    global_work_size[1] =
        ((size_t)image_height + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1];

    // 7. 計算 local memory 大小，傳給 __local float *tile（kernel 第 6 個參數）
    int half = filter_width / 2;
    size_t tile_w = local_work_size[0] + 2 * (size_t)half;
    size_t tile_h = local_work_size[1] + 2 * (size_t)half;
    size_t local_mem_size = tile_w * tile_h * sizeof(float);

    status |= clSetKernelArg(kernel, 6, local_mem_size, NULL);

    // 8. 啟動 kernel
    status = clEnqueueNDRangeKernel(queue,
                                    kernel,
                                    2,
                                    NULL,
                                    global_work_size,
                                    local_work_size,
                                    0, NULL, NULL);

    // 9. 等待 kernel 結束，並讀回結果
    status = clEnqueueReadBuffer(queue,
                                 d_output,
                                 CL_TRUE,
                                 0,
                                 image_size,
                                 output_image,
                                 0,
                                 NULL,
                                 NULL);

    // 10. 釋放 OpenCL 資源
    clReleaseKernel(kernel);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_filter);
    clReleaseCommandQueue(queue);
}
