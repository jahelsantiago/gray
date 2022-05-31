
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>

#include "stb_image.h"
#include "stb_image_write.h"

struct  Pixel
{
    unsigned char r, g, b, a;
};


void ConvertImageToGrayCpu(unsigned char* image, int width, int height)
{
    for (int row = 0; row < width; row++)
    {
        for (int col = 0; col < height; col++)
        {
            Pixel* pixel = (Pixel*)&image[(row * width + col) * 4];
            unsigned char pixle_gray = (unsigned char)(0.299f * pixel->r + 0.587f * pixel->g + 0.114f * pixel->b);
            pixel -> r = pixel -> g = pixel -> b = pixle_gray;
            pixel -> a = 255;
        } 
        
    }
}

__global__ void ConvertImageToGrayCuda(unsigned char* image, int width, int height)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < width && col < height)
    {
        Pixel* pixel = (Pixel*)&image[(row * width + col) * 4];
        unsigned char pixle_gray = (unsigned char)(0.299f * pixel->r + 0.587f * pixel->g + 0.114f * pixel->b);
        pixel -> r = pixel -> g = pixel -> b = pixle_gray;
        pixel -> a = 255;
    }
}

int main() {

    //path to the image
    char* path = "lena.png";

    // Load image
    int width, height, channels;
    unsigned char* imageData = stbi_load(path, &width, &height, &channels, 4);
    //print the first pixel
    if(imageData == nullptr) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }

    // validate image size 
    if(width % 32 || height % 32) {
        std::cerr << "Image size must be multiple of 32" << std::endl;
        return -1;
    }

    // convert image to gray scale with Cpu --------------------------------------------------
    std::cout << "Processing on cpu...";
    //ConvertImageToGrayCpu(imageData, width, height);
    std::cout << "Done" << std::endl;

    // convert image to gray scale with Cuda --------------------------------------------------

    // allocate memory for the image on the device
    std::cout << "Copying date to device ...";
    unsigned char* imageDataDevice = nullptr;
    cudaMalloc(&imageDataDevice, width * height * 4);
    assert(imageDataDevice != nullptr);
    // copy the image to the device
    cudaMemcpy(imageDataDevice, imageData, width * height * 4, cudaMemcpyHostToDevice);
    std::cout << "Done" << std::endl;

    // process the image on the device
    std::cout << "Processing on device...";
    dim3 block(32, 32);
    dim3 grid(width / 32, height / 32);
    ConvertImageToGrayCuda<<<grid, block>>>(imageDataDevice, width, height);
    std::cout << "Done" << std::endl;

    // copy the image back to the host
    std::cout << "Copying data back to host...";
    cudaMemcpy(imageData, imageDataDevice, width * height * 4, cudaMemcpyDeviceToHost);
    std::cout << "Done" << std::endl;


    //write image back to disk
    std::cout << "Writing png to disk...";
    stbi_write_png("lena_out.png", width, height, 4, imageData, 4*width);
    std::cout << "Done" << std::endl;

    //close image
    stbi_image_free(imageData);

    return 0;
}


