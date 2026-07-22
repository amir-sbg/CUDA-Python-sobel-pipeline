extern "C" __global__ void sobel_edges(
    const float* image,
    float* output,
    int height,
    int width) {
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
        -image[(y - 1) * width + (x - 1)] + image[(y - 1) * width + (x + 1)]
        - 2.0F * image[y * width + (x - 1)] + 2.0F * image[y * width + (x + 1)]
        - image[(y + 1) * width + (x - 1)] + image[(y + 1) * width + (x + 1)];
    const float vertical =
        -image[(y - 1) * width + (x - 1)] - 2.0F * image[(y - 1) * width + x]
        - image[(y - 1) * width + (x + 1)] + image[(y + 1) * width + (x - 1)]
        + 2.0F * image[(y + 1) * width + x] + image[(y + 1) * width + (x + 1)];
    output[index] = sqrtf(horizontal * horizontal + vertical * vertical);
}
