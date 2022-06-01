
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

#include <iostream>
#include <string>
#include <cassert>

#include "stb_image.h"
#include "stb_image_write.h"

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

__global__ void ConvertImageToGrayCuda(unsigned char *image, int width, int height)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i <= 100; i++)
    {
        for (int j = 0; j <= 100; j++)
        {
            if (row < width && col < height)
            {
                Pixel *pixel = (Pixel *)&image[((row * width + col) + (i * blockDim.x) + (j * blockDim.y)) * 4];
                unsigned char pixle_gray = (unsigned char)(0.299f * pixel->r + 0.587f * pixel->g + 0.114f * pixel->b);
                pixel->r = pixel->g = pixel->b = pixle_gray;
                pixel->a = 255;
            }
        }
    }
}

__global__ void ConvertImageToGrayLinear(unsigned char *image, int width, int height)
{
    int pixel_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixel_index < width * height)
    {
        Pixel *pixel = (Pixel *)&image[pixel_index * 4];
        unsigned char pixle_gray = (unsigned char)(0.299f * pixel->r + 0.587f * pixel->g + 0.114f * pixel->b);
        pixel->r = pixel->g = pixel->b = pixle_gray;
        pixel->a = 255;
    }
}

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

int main()
{

    // path to the image
    char *path = "lena.png";

    // Load image
    int width, height, channels;
    unsigned char *imageData = stbi_load(path, &width, &height, &channels, 4);

    // print the value of the width, height and channels
    std::cout << "width: " << width << std::endl;
    std::cout << "height: " << height << std::endl;
    std::cout << "channels: " << channels << std::endl;

    if (imageData == nullptr)
    {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }

    // validate image size
    if (width % 32 || height % 32)
    {
        std::cerr << "Image size must be multiple of 32" << std::endl;
        return -1;
    }

    // runOnCpu(imageData, width, height);

    // convert image to gray scale with Cuda --------------------------------------------------

    // allocate memory for the image on the device
    std::cout << "Copying date to device ...";
    unsigned char *imageDataDevice = nullptr;
    cudaMalloc(&imageDataDevice, width * height * 4);
    assert(imageDataDevice != nullptr);
    // copy the image to the device
    cudaMemcpy(imageDataDevice, imageData, width * height * 4, cudaMemcpyHostToDevice);
    std::cout << "Done" << std::endl;

    // process the image on the device
    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Processing on device...";

    ConvertImageToGrayLinear<<<102400, 1024>>>(imageDataDevice, width, height);
    std::cout << "Done" << std::endl;
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_gpu = end_gpu - start_gpu;
    std::cout << "Time taken by function on Device: " << elapsed_gpu.count() << " seconds" << std::endl;

    // copy the image back to the host
    std::cout << "Copying data back to host...";
    cudaMemcpy(imageData, imageDataDevice, width * height * 4, cudaMemcpyDeviceToHost);
    std::cout << "Done" << std::endl;

    // write image back to disk
    std::cout << "Writing png to disk...";
    stbi_write_png("lena_out.png", width, height, 4, imageData, 4 * width);
    std::cout << "Done" << std::endl;

    // close image
    stbi_image_free(imageData);

    return 0;
}
