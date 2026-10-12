extern "C" __global__ void sobel_edges(
    const float* image,
    float* output,
    int height,
    int width,
    float corner_weight,
    float axial_weight) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int index = y * width + x;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        output[index] = 0.0F;
        return;
    }

    const float horizontal =
        -corner_weight * image[(y - 1) * width + (x - 1)]
        + corner_weight * image[(y - 1) * width + (x + 1)]
        - axial_weight * image[y * width + (x - 1)]
        + axial_weight * image[y * width + (x + 1)]
        - corner_weight * image[(y + 1) * width + (x - 1)]
        + corner_weight * image[(y + 1) * width + (x + 1)];
    const float vertical =
        -corner_weight * image[(y - 1) * width + (x - 1)]
        - axial_weight * image[(y - 1) * width + x]
        - corner_weight * image[(y - 1) * width + (x + 1)]
        + corner_weight * image[(y + 1) * width + (x - 1)]
        + axial_weight * image[(y + 1) * width + x]
        + corner_weight * image[(y + 1) * width + (x + 1)];
    output[index] = sqrtf(horizontal * horizontal + vertical * vertical);
}
