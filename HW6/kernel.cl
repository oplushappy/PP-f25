#define TILE_SIZE 16

// 告訴編譯器我們固定用 16x16
__kernel __attribute__((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1)))
void convolution(__global const float * restrict input_image,  // 加入 restrict
                 __global float * restrict output_image,       // 加入 restrict
                 __constant float * restrict filter,           // 加入 restrict
                 const int image_width,
                 const int image_height,
                 const int filter_width,
                 __local float *local_buffer)
{
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);

    const int halffilter_size = filter_width / 2;
    const int buffer_w = TILE_SIZE + filter_width - 1;
    const int group_start_x = get_group_id(0) * TILE_SIZE - halffilter_size;
    const int group_start_y = get_group_id(1) * TILE_SIZE - halffilter_size;

    // phase 1: 合作搬運 (Global -> Local)
    const int total_pixels = buffer_w * buffer_w;
    const int thread_id_flat = ty * TILE_SIZE + tx;
    const int num_threads = TILE_SIZE * TILE_SIZE;

    for (int i = thread_id_flat; i < total_pixels; i += num_threads) {
        int ly = i / buffer_w;
        int lx = i % buffer_w;
        
        int input_y = group_start_y + ly;
        int input_x = group_start_x + lx;

        float val = 0.0f;
        if (input_y >= 0 && input_y < image_height && input_x >= 0 && input_x < image_width) {
             val = input_image[input_y * image_width + input_x];
        }
        local_buffer[i] = val; // 直接用 i (即 ly*buffer_w + lx) 寫入
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // phase 2: convulution

    if (gy < image_height && gx < image_width) {
        float sum = 0.0f;
        
        // 預先計算當前 Thread 在 Local Buffer 的起始位置
        // 原本公式: (ty + k) * buffer_w + (tx + l)
        // 拆解: (ty * buffer_w + tx) + (k * buffer_w + l)
        // 先算好 base_offset = ty * buffer_w + tx
        int base_offset = ty * buffer_w + tx;

        #pragma unroll
        for (int k = 0; k < filter_width; k++) {
            
            // 預先算出這一列 Filter 在 Buffer 的偏移量
            int row_offset = base_offset + (k * buffer_w);
            int filter_row_idx = k * filter_width;

            #pragma unroll
            for (int l = 0; l < filter_width; l++) {
                sum += local_buffer[row_offset + l] * filter[filter_row_idx + l];
            }
        }
        
        output_image[gy * image_width + gx] = sum;
    }
}