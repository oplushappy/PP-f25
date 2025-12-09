__kernel void convolution(__global const float *input_image,
                          __global float *output_image,
                          __global const float *filter,
                          const int image_width,
                          const int image_height,
                          const int filter_width,
                          __local float *tile)
{
    // work-group / work-item 資訊
    int lj = get_local_id(0);      // local x (column inside tile)
    int li = get_local_id(1);      // local y (row inside tile)
    int lw = get_local_size(0);    // tile 寬度，例如 16
    int lh = get_local_size(1);    // tile 高度，例如 16

    int group_j = get_group_id(0); // work-group 在 X 方向的 index
    int group_i = get_group_id(1); // work-group 在 Y 方向的 index

    int half = filter_width / 2;

    // 這個 work-group 對應的 output tile 左上角在 global image 的位置
    int base_j = group_j * lw;
    int base_i = group_i * lh;

    // local tile 尺寸 = work-group 尺寸 + halo（filter 外圍）
    int tile_w = lw + 2 * half;
    int tile_h = lh + 2 * half;

    int ty;
    int tx;
    int img_i;
    int img_j;

    // ===============================
    // 1. 所有 work-items 合作，把 tile + halo 從 global memory 載到 local memory (tile[])
    // ===============================
    ty = li;
    while (ty < tile_h)
    {
        img_i = base_i + ty - half; // 對應到 global image 的 row

        tx = lj;
        while (tx < tile_w)
        {
            img_j = base_j + tx - half; // 對應到 global image 的 col

            float val = 0.0f;

            // 邊界檢查：超出圖片範圍 → 當 0（zero padding）
            if (img_i >= 0 && img_i < image_height &&
                img_j >= 0 && img_j < image_width)
            {
                val = input_image[img_i * image_width + img_j];
            }

            tile[ty * tile_w + tx] = val;

            tx += lw; // 同一 row 往右跨 tile 寬度跳
        }

        ty += lh;     // 下一個 row，由不同 work-item 負責
    }

    // 等所有 work-item 都把 tile[] 填完
    barrier(CLK_LOCAL_MEM_FENCE);

    // ===============================
    // 2. 每個 work-item 用 tile[] 做卷積，算自己負責的 output pixel
    // ===============================

    int j = base_j + lj;   // global column
    int i = base_i + li;   // global row

    // 超出實際影像範圍的 padding 區域就不用算
    if (i >= image_height || j >= image_width)
    {
        return;
    }

    float sum = 0.0f;

    // 在 tile[] 裡，對應到這個 output pixel 的「中心」位置
    int center_i = li + half;
    int center_j = lj + half;

    int k;
    int l;
    int fk;
    int fl;
    int ti;
    int tj;
    float filter_val;
    float image_val;

    k = -half;
    while (k <= half)
    {
        l = -half;
        while (l <= half)
        {
            fk = k + half;
            fl = l + half;

            // filter 的值（仍然從 global constant filter 拿）
            filter_val = filter[fk * filter_width + fl];

            // local tile 中對應的 input 像素位置
            ti = center_i + k;
            tj = center_j + l;

            image_val = tile[ti * tile_w + tj];

            sum += image_val * filter_val;

            l++;
        }
        k++;
    }

    // 寫回 global output
    output_image[i * image_width + j] = sum;
}
