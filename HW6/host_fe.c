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
    // cl_int status;
    // int filter_size = filter_width * filter_width;
    cl_int status;

    size_t image_size  = (size_t)image_height * (size_t)image_width * sizeof(float);
    size_t filter_size = (size_t)filter_width * (size_t)filter_width * sizeof(float);

    // 1. 建立 command queue
    // cl_command_queue clCreateCommandQueue( cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int* errcode_ret);
    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &status);
    // CHECK(status, "clCreateCommandQueue");

    // 2. 建立 device buffer
    // need 3 buffer : input_image、filter、output_image
    // cl_mem clCreateBuffer( cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret);
    cl_mem d_input = clCreateBuffer(*context, CL_MEM_READ_ONLY, image_size, NULL, &status);
    // CHECK(status, "clCreateBuffer(input)");

    cl_mem d_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, image_size, NULL, &status);
    // CHECK(status, "clCreateBuffer(output)");

    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, filter_size, NULL, &status);
    // CHECK(status, "clCreateBuffer(filter)");

    // 3. 將 host 資料寫入 device
    // cl_int clEnqueueWriteBuffer( cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
    status = clEnqueueWriteBuffer(queue, d_input, CL_FALSE, 0, image_size, input_image, 0, NULL, NULL);
    // CHECK(status, "clEnqueueWriteBuffer(input)");

    status = clEnqueueWriteBuffer(queue, d_filter, CL_FALSE, 0, filter_size, filter, 0, NULL, NULL);
    // CHECK(status, "clEnqueueWriteBuffer(filter)");

    // 4. 建立 kernel
    // cl_kernel clCreateKernel( cl_program program, const char* kernel_name, cl_int* errcode_ret);
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    // CHECK(status, "clCreateKernel(convolution)");

    // 5. 設定 kernel 參數
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

    // 6. 啟動 2D NDRange：每個 work-item 負責一個像素 (j, i)
    size_t global_work_size[2];
    global_work_size[0] = (size_t)image_width;   // x / column (j)
    global_work_size[1] = (size_t)image_height;  // y / row (i)

    // cl_int clEnqueueNDRangeKernel( cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    // CHECK(status, "clEnqueueNDRangeKernel");

    // 7. 等待 kernel 結束，並讀回結果
    // cl_int clEnqueueReadBuffer( cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
    status = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, image_size, output_image, 0, NULL, NULL);
    // CHECK(status, "clEnqueueReadBuffer(output)");

    // 8. 釋放 OpenCL 資源（context/program 由外面管理，不在這裡釋放）
    clReleaseKernel(kernel);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_filter);
    clReleaseCommandQueue(queue);
}
