#define TILE_SIZE 16

__kernel void convolution(__global const float *input_image,
                          __global float *output_image,
                          __constant float *filter,
                          const int image_width,
                          const int image_height,
                          const int filter_width,
                          __local float *local_buffer)
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    int halffilter_size = filter_width / 2;
    int buffer_w = TILE_SIZE + filter_width - 1; // Local Buffer 的真實寬度
    int total_pixels = buffer_w * buffer_w;      // 需要搬運的總像素數

    // 計算 WorkGroup 對應的原圖左上角
    int group_start_x = get_group_id(0) * TILE_SIZE - halffilter_size;
    int group_start_y = get_group_id(1) * TILE_SIZE - halffilter_size;

    // global -> local
    
    // 將 2D Thread ID 攤平成 1D，方便分配搬運工作
    int thread_id_flat = ty * TILE_SIZE + tx;
    int num_threads = TILE_SIZE * TILE_SIZE;

    // 迴圈搬運
    for (int i = thread_id_flat; i < total_pixels; i += num_threads) {
        // 將 1D 索引轉回 Local Buffer 的 2D 座標
        int ly = i / buffer_w;
        int lx = i % buffer_w;
        
        int input_y = group_start_y + ly;
        int input_x = group_start_x + lx;

        // 寫入 Local Buffer (注意：使用 1D 索引 ly * buffer_w + lx)
        if (input_y >= 0 && input_y < image_height && input_x >= 0 && input_x < image_width) {
            local_buffer[ly * buffer_w + lx] = input_image[input_y * image_width + input_x];
        } else {
            local_buffer[ly * buffer_w + lx] = 0.0f; // Padding
        }
    }

    // 等待搬運完成
    barrier(CLK_LOCAL_MEM_FENCE);

    // Convulution

    if (gy < image_height && gx < image_width) {
        float sum = 0.0f;
        
        for (int k = 0; k < filter_width; k++) {
            for (int l = 0; l < filter_width; l++) {
                // 計算在 Local Buffer 中的位置
                // tx 對應的是 buffer 中間的區域，所以要加上 k, l 即可
                int buffer_y = ty + k;
                int buffer_x = tx + l;
                
                // 讀取 Local Buffer (使用 1D 索引)
                sum += local_buffer[buffer_y * buffer_w + buffer_x] * filter[k * filter_width + l];
            }
        }
        
        output_image[gy * image_width + gx] = sum;
    }
}