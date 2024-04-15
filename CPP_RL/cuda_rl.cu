#include <iostream>
#include <cuda.h>
#include <cudnn.h>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include "cuda_rl.cuh"
#include "algorithms.cuh"
#include "fftconv.cuh"
#include "file_io.cuh"

namespace fs = std:filesystem;
typedef float2 Complex;

int main(int argc, char** argv)
{
    cudaError_t error;

    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Number of GPUs: " << numGPUs << std::endl;

        // Use all available GPUs
    for (int i = 0; i < numGPUs; i++)
    {
        cudaSetDevice(i);
        cudaDeviceEnablePeerAccess(i, 0);
    }

    // Search for all the images in the directory
    std::vector<std::string> imgPaths;
    std::vector<std::string> kernPaths;
    std::string imgPath = "C:/some/path/to/images/";
    std::string kernPath = "C:/some/path/to/kernels/";

    imgPaths = fileSearch(imgPath, ".tif");
    int numImages = imgPaths.size();

    kernPaths = fileSearch(kernPath, ".tif");
    int numKernels = kernPaths.size();
    
    // Find the size of the image and kernel
    ImageData testimg = readImage(imgPaths[0]);
    ImageData testkern = readImage(kernPaths[0]);

    imgx = testimg.width;
    imgy = testimg.height;
    kernx = testkern.width;
    kerny = testkern.height;

    // Allocate memory for the image and kernel data
    float* imgdevptr;
    float* kerndevptr;
    float* result2ddevptr;
    float* result3ddevptr;

    cudaError_t imgmem, kernmem, resultmem;

    imgmem = cudaMalloc((void**)&imgdevptr, imgx * imgy * sizeof(float)); // Should only store one image at a time for memory efficiency
    kernmem = cudaMalloc((void**)&kerndevptr, kernx * kerny * numKernels * sizeof(float)); // NEED to store all the kernels at once
    resultmem = cudaMalloc((void**)&result2ddevptr, imgx * imgy * sizeof(float)); // Should only store one image at a time for memory efficiency
    resultmem = cudaMalloc((void**)&result3ddevptr, imgx * imgy * numKernels * sizeof(float)); // NEED to store all the results at once

    // Copy kernel data to device
    for (int i = 0; i < numKernels; i++)
    {
        ImageData kerndata = readImage(kernPaths[i]);
        cudaMemcpy(kerndevptr + i * kernx * kerny, kerndata.data, kernx * kerny * sizeof(float), cudaMemcpyHostToDevice); 
    }

    if (kernmem != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory for the kernel on the device" << std::endl;
        exit(1);
    }

    float* backkerndevptr;

    cudaError_t backkernmem;

    backkernmem = cudaMalloc((void**)&backkerndevptr, kernx * kerny * numKernels * sizeof(float)); // NEED to store all the kernels at once

    // Copy kernel data to device, but backwards
    for (int i = 0; i < numKernels; i++)
    {
        ImageData kerndata = readImage(kernPaths[i]);
        std::reverse(kerndata.data.begin(), kerndata.data.end());
        cudaMemcpy(backkerndevptr + i * kernx * kerny, kerndata.data, kernx * kerny * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (backkernmem != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory for the kernel on the device" << std::endl;
        exit(1);
    }

    thrust::device_vector<float> kernsum(kernx * kerny); // Will be used to normalize the kernel

    // Normalize the forward kernel
    for (int i = 0; i < numKernels; i++)
    {
        float forward_sum = thrust::reduce(kerndevptr + i * kernx * kerny, kerndevptr + (i + 1) * kernx * kerny);
        forward_sum += 1e-6; // Add a small number to avoid division by zero
        thrust::device_vector<float> kernvec = thurst::device_vector<float>(kerndevptr + i * kernx * kerny, kerndevptr + (i + 1) * kernx * kerny);
        thurst::transform(kernvec.begin(), kernvec.end(), thrust::make_constant_iterator(forward_sum), kernvec.begin(), thurst::divide<float>());

        thrust::transform(backkernvec.begin(), backkernvec.end(), kernsum.begin(), kernsum.begin(), thurst::add<float>());

        // Copy the kernel back to the device
        cudaMemcpy(kerndevptr + i * kernx * kerny, kernvec.data(), kernx * kerny * sizeof(float), cudaMemcpyHostToDevice);
        
        // Don't forget to clear the vector
        kernvec.clear();
        kernvec.shrink_to_fit();
    }

    // Normalize the kernels
    thrust::transform(kernsum.begin(), kernsum.end(), thrust::make_constant_iterator(numKernels), kernsum.begin(), thurst::multiply<float>());
    thrust::transform(kernsum.begin(), kernsum.end(), thrust::make_constant_iterator(1e-6), kernsum.begin(), thurst::add<float>());

    // Make the backward kernel
    for (int i = 0; i < numKernels; i++)
    {
        thrust::device_vector<float> backkern = thurst::device_vector<float>(backkerndevptr + i * kernx * kerny, backkerndevptr + (i + 1) * kernx * kerny);
        thrust::transform(backkern.begin(), backkern.end(), kernsum.begin(), backkern.begin(), thurst::divide<float>());
        cudaMemcpy(backkerndevptr + i * kernx * kerny, backkern.data(), kernx * kerny * sizeof(float), cudaMemcpyHostToDevice);

        backkern.clear();
        backkern.shrink_to_fit();
    }
    // Clear the kernel sum
    kernsum.clear();
    kernsum.shrink_to_fit();

    /// Read images sequentially
    for (i = 0, i < Images, i++)
    {
        std::vector<float>* imgPtr = readImage(imgPaths[i]);
        if (imgPtr != nullptr)
        {
            std::cout << "Image read successfully" << std::endl;
            float* imgPinnedMem; // Pinned memory for image data
            size_t imgByteSize = imgPtr->size() * sizeof(float); // Size of the image in bytes
            cudaError_t error = cudaHostAlloc((void**)&imgPinnedMem, imgByteSize); // Allocate pinned memory for image data

            // Copy data from the vector to pinned memory
            cudaMemcpy(imgPinnedMem, imgPtr->data(), imgByteSize); // Copy the image data to pinned memory

            if (error != cudaSuccess) 
            {
                std::cerr << "Failed to allocate pinned memory for image data" << std::endl;
                //delete ImagePtr; // Release ImagePtr if allocation fails
            }

        std::cout << "Image size is: " << imgPtr->size() << std::endl;
        std::cout << "Kernel size is: " << kernPtr->size() << std::endl;

        if (imgPtr->size() != kernPtr->size())
            {
            std::cerr << "Image and kernel sizes do not match" << std::endl;
            exit(2);
            }
        }

        // Continue with the algorithm
        rlAlg(imgdevptr, kerndevptr, backkerndevptr, result2ddevptr, result3ddevptr, imgPtr->size(), kernPtr->size(), numKernels, 400);
    }
}