// 定義 Tile 大小，必須與 host_fe.c 中的 local_work_size (16) 一致
#define TILE_SIZE 16

// 定義 Local Buffer 的最大寬度
// 公式: TILE_SIZE + (MAX_FILTER_WIDTH - 1)
// 假設最大 Filter 為 17x17 (半徑8)，Buffer 需要 16 + 16 = 32
#define BUFFER_SIZE 32

__kernel void convolution(__global const float *input_image,
                          __global float *output_image,
                          __constant float *filter, // 優化: 放在 Constant Memory
                          const int image_width,
                          const int image_height,
                          const int filter_width)
{
    // 宣告 Local Memory (Shared Memory)
    // 每個 WorkGroup 共用這塊記憶體
    __local float local_buffer[BUFFER_SIZE][BUFFER_SIZE];

    // 取得各種 ID
    // tx, ty: 在 Group 內的相對座標 (0 ~ 15)
    int tx = get_local_id(0); 
    int ty = get_local_id(1);

    // gx, gy: 在整張影像上的絕對座標
    int gx = get_global_id(0); 
    int gy = get_global_id(1);

    // Filter 半徑
    int halffilter_size = filter_width / 2;

    // 計算這個 WorkGroup 負責的 Input Tile 左上角座標
    // 注意：Input Tile 比 Output Tile (16x16) 還要大，因為包含周圍的 Halo
    int group_start_x = get_group_id(0) * TILE_SIZE - halffilter_size;
    int group_start_y = get_group_id(1) * TILE_SIZE - halffilter_size;

    // ==========================================================
    // 階段 1: 合作將資料從 Global Memory 搬運到 Local Memory
    // ==========================================================
    
    // 計算 Local Buffer 實際需要的寬度 (Tile + Halo)
    int buffer_w = TILE_SIZE + filter_width - 1;
    int total_pixels_to_load = buffer_w * buffer_w;
    
    // 將 2D Thread ID 攤平成 1D ID (0 ~ 255)
    int thread_id_flat = ty * TILE_SIZE + tx;
    int num_threads = TILE_SIZE * TILE_SIZE; // 256

    // 每個 Thread 可能需要搬運多個像素 (因為 buffer_w^2 > 256)
    for (int i = thread_id_flat; i < total_pixels_to_load; i += num_threads) {
        int ly = i / buffer_w; // Buffer 內的 row
        int lx = i % buffer_w; // Buffer 內的 col
        
        int input_y = group_start_y + ly; // 原圖 row
        int input_x = group_start_x + lx; // 原圖 col

        // 邊界檢查與 Zero Padding
        if (input_y >= 0 && input_y < image_height && input_x >= 0 && input_x < image_width) {
            local_buffer[ly][lx] = input_image[input_y * image_width + input_x];
        } else {
            local_buffer[ly][lx] = 0.0f;
        }
    }

    // 等待所有 Thread 搬運完成
    barrier(CLK_LOCAL_MEM_FENCE);

    // ==========================================================
    // 階段 2: 進行卷積運算 (只讀取快速的 Local Memory)
    // ==========================================================

    // 只有在影像範圍內的 Thread 才需要計算並寫入
    if (gy < image_height && gx < image_width) {
        float sum = 0.0f;
        
        // 掃描 Filter
        // 注意：local_buffer 的索引方式
        // 當 k=0, l=0 時，對應到 filter 左上角
        // 對於 thread (tx, ty)，其對應的 input 資料起始點在 local_buffer[ty][tx]
        // 所以直接加上 k, l 即可
        for (int k = 0; k < filter_width; k++) {
            for (int l = 0; l < filter_width; l++) {
                sum += local_buffer[ty + k][tx + l] * filter[k * filter_width + l];
            }
        }
        
        output_image[gy * image_width + gx] = sum;
    }
}