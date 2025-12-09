__kernel void convolution(__global const float *input_image,
                          __global float *output_image,
                          __global const float *filter,
                          const int image_width,
                          const int image_height,
                          const int filter_width)
{
    // j = column (x), i = row (y)
    int j = get_global_id(0);  // x
    int i = get_global_id(1);  // y

    // 避免 global size 大於影像尺寸
    if (i >= image_height || j >= image_width) return;

    int halffilter_size = filter_width / 2;
    float sum = 0.0f;

    // 針對 pixel (i, j) 掃過 filter 視窗
    for (int k = -halffilter_size; k <= halffilter_size; k++)
    {
        for (int l = -halffilter_size; l <= halffilter_size; l++)
        {
            int ii = i + k;  // 原圖 row
            int jj = j + l;  // 原圖 col

            // Zero padding：超出邊界一律忽略（當 0）
            if (ii >= 0 && ii < image_height && jj >= 0 && jj < image_width) {
                float image_val = input_image[ii * image_width + jj];

                int fk = k + halffilter_size;
                int fl = l + halffilter_size;

                float filter_val = filter[fk * filter_width + fl];

                sum += image_val * filter_val;
            }
        }
    }

    // 寫回輸出影像
    output_image[i * image_width + j] = sum;
}
