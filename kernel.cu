#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

#include <iostream>
#include <string>
#include <cassert>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

struct Pixel
{
    unsigned char r, g, b, a;
};

void getDeviceCharacteristics()
{
    // get the caracteristics of the device
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // print the value of the deviceProp
    std::cout << "deviceProp.name: " << deviceProp.name << std::endl;
    std::cout << "deviceProp.major: " << deviceProp.major << std::endl;
    std::cout << "deviceProp.minor: " << deviceProp.minor << std::endl;
    std::cout << "deviceProp.multiProcessorCount: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "deviceProp.totalGlobalMem: " << deviceProp.totalGlobalMem << std::endl;
    std::cout << "deviceProp.sharedMemPerBlock: " << deviceProp.sharedMemPerBlock << std::endl;
    std::cout << "deviceProp.regsPerBlock: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "deviceProp.warpSize: " << deviceProp.warpSize << std::endl;
    std::cout << "deviceProp.memPitch: " << deviceProp.memPitch << std::endl;
    std::cout << "deviceProp.maxThreadsPerBlock: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "deviceProp.maxThreadsDim[0]: " << deviceProp.maxThreadsDim[0] << std::endl;
    std::cout << "deviceProp.maxThreadsDim[1]: " << deviceProp.maxThreadsDim[1] << std::endl;
    std::cout << "deviceProp.maxThreadsDim[2]: " << deviceProp.maxThreadsDim[2] << std::endl;
    std::cout << "deviceProp.maxGridSize[0]: " << deviceProp.maxGridSize[0] << std::endl;
    std::cout << "deviceProp.maxGridSize[1]: " << deviceProp.maxGridSize[1] << std::endl;
    std::cout << "deviceProp.maxGridSize[2]: " << deviceProp.maxGridSize[2] << std::endl;
    std::cout << "deviceProp.totalConstMem: " << deviceProp.totalConstMem << std::endl;
    std::cout << "deviceProp.major: " << deviceProp.major << std::endl;
    std::cout << "deviceProp.minor: " << deviceProp.minor << std::endl;
    std::cout << "deviceProp.textureAlignment: " << deviceProp.textureAlignment << std::endl;
    std::cout << "deviceProp.texturePitchAlignment: " << deviceProp.texturePitchAlignment << std::endl;
    std::cout << "deviceProp.deviceOverlap: " << deviceProp.deviceOverlap << std::endl;
    std::cout << "deviceProp.multiProcessorCount: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "deviceProp.kernelExecTimeoutEnabled: " << deviceProp.kernelExecTimeoutEnabled << std::endl;
    std::cout << "deviceProp.integrated: " << deviceProp.integrated << std::endl;
    std::cout << "deviceProp.canMapHostMemory: " << deviceProp.canMapHostMemory << std::endl;
    std::cout << "deviceProp.computeMode: " << deviceProp.computeMode << std::endl;
}

/**
 * @brief      { Converts the image to gray using CUDA }
 *
 * @param      image  The image
 * @param      width  The width of the image in pixels
 * @param      height The height of the image in pixels
 * @return
 */
void ConvertImageToGrayCpu(unsigned char *image, int width, int height)
{
    for (int row = 0; row < width; row++)
    {
        for (int col = 0; col < height; col++)
        {
            Pixel *pixel = (Pixel *)&image[(row * width + col) * 4];
            unsigned char pixle_gray = (unsigned char)(0.299f * pixel->r + 0.587f * pixel->g + 0.114f * pixel->b);
            pixel->r = pixel->g = pixel->b = pixle_gray;
            pixel->a = 255;
        }
    }
}

/**
 * @brief      { Converts the image to gray using CUDA }
 *
 * @param      image  The image
 * @param      width  The width of the image in pixels
 * @param      height The height of the image in pixels
 * @param      iterations  The number of iterations that the kernel will run
 */
__global__ void ConvertImageToGrayLinear(unsigned char *image, int width, int height, int iterations)
{
    int pixel_index = threadIdx.x + blockIdx.x * blockDim.x;
    // get the total number of threads
    int total_threads = blockDim.x * gridDim.x;
    for (int i = 0; i < iterations; i++)
    {
        if (pixel_index < width * height)
        {
            Pixel *pixel = (Pixel *)&image[pixel_index * 4];
            unsigned char pixle_gray = (unsigned char)(0.299f * pixel->r + 0.587f * pixel->g + 0.114f * pixel->b);
            pixel->r = pixel->g = pixel->b = pixle_gray;
            pixel->a = 255;
        }
        pixel_index += total_threads;
    }
}

/**
 * @brief Run the algorithm on the CPU
 *
 * @param image - the image to be converted
 * @param width - the width of the image
 * @param height - the height of the image
 */
void runOnCpu(unsigned char *imageData, int width, int height)
{
    auto start = std::chrono::high_resolution_clock::now();
    // convert image to gray scale with Cpu --------------------------------------------------
    std::cout << "Processing on cpu...";
    ConvertImageToGrayCpu(imageData, width, height);
    std::cout << "Done" << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken by function on Cpu: " << elapsed.count() << " seconds" << std::endl;
}

/**
 * @brief Run the algorithm on the GPU
 *
 * @param imageData - image data to be processed on GPU device (CUDA)
 * @param width - image width in pixels
 * @param height - image height in pixels
 */
void runOnDevice(unsigned char *imageData, int width, int height, int blocks, int threads_per_block)
{
    // allocate memory for the image on the device
    unsigned char *imageDataDevice = nullptr;
    cudaMalloc(&imageDataDevice, width * height * 4);
    assert(imageDataDevice != nullptr);
    // copy the image to the device
    cudaMemcpy(imageDataDevice, imageData, width * height * 4, cudaMemcpyHostToDevice);

    // process the image on the device
    auto start_gpu = std::chrono::high_resolution_clock::now();

    int total_pixel = width * height;
    int iterations = total_pixel / (blocks * threads_per_block);
    if (total_pixel % (blocks * threads_per_block) != 0)
        iterations++;

    ConvertImageToGrayLinear<<<blocks, threads_per_block>>>(imageDataDevice, width, height, iterations);
    cudaDeviceSynchronize();

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_gpu = end_gpu - start_gpu;
    std::cout << "Time taken by function on Device: " << elapsed_gpu.count() << " seconds" << std::endl;

    // copy the image back to the host
    cudaMemcpy(imageData, imageDataDevice, width * height * 4, cudaMemcpyDeviceToHost);
    // free the device memory
    cudaFree(imageDataDevice);
}

/**
 * @brief      { Main function }
 *
 * @param      argc  The argc
 * @param      argv  The argv.
 * The first argument is the path to the image file,
 * the second argument is the output path for the converted image
 * the third argument is the number of blocks
 * the fourth argument is the number of threads per block
 *
 * @return     { int }
 */
int main(int argc, char *argv[])
{

    if (argc != 5)
    {
        std::cout << "Usage: " << argv[0] << " <image_path> <output_path> <blocks> <threads_per_block>" << std::endl;
        return -1;
    }

    char *path = argv[1];
    char *output_path = argv[2];
    int blocks = atoi(argv[3]);
    int threads_per_block = atoi(argv[4]);

    // Load image
    int width, height, channels;
    unsigned char *imageData = stbi_load(path, &width, &height, &channels, 4);
    if (imageData == nullptr)
    {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }

    // runOnCpu(imageData, width, height);
    runOnDevice(imageData, width, height, blocks, threads_per_block);

    // write image back to disk
    stbi_write_png(output_path, width, height, 4, imageData, 4 * width);

    // free memory
    stbi_image_free(imageData);

    return 0;
}
