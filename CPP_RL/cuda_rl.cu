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

    // Search for all the images in the directory
    std::vector<std::string> imgPaths;
    std::vector<std::string> kernPaths;

    std::string imgPath;
    std::string kernPath;

    std::cout << "Enter the path to the images: " << std::endl;
    std::cin >> imgPath;
    std::cout << "Enter the path to the kernels: " << std::endl;
    std::cin >> kernPath;

    imgPaths = fileSearch(imgPath, ".tif");
    int numImages = imgPaths.size();
    std::cout << "Number of images: " << numImages << std::endl;

    kernPaths = fileSearch(kernPath, ".tif");
    int numKernels = kernPaths.size();
    std::cout << "Number of kernels: " << numKernels << std::endl;
    
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

    psfNorm(); // Normalize the kernel
    psfbNorm(); // Normalize the backward kernel
    fftpsf(); // Take the FFT of the kernel

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
    }
    /// Load PSF and PSFb into device memory and normalize
    psfNorm();
    psfbNorm();

    /// Take the FFT of the kernel
    fftpsf();
    fftpsf();

    /// Ask the user for the number of iterations
    int itr;
    std::cout << "Enter the number of iterations: ";
    std::cin >> itr;

    str resultPath = "C:/some/path/to/results/";
    std::cout << "Enter path to save the result: ";
    std::cin >> resultPath;

    /// Run the RL algorithm
    thrust::host_vector<float>> result_3d(imgx * imgy * numKern);
    for i in range(imgPaths.size())
    {
        time_t start, end;
        time(&start);
        rlAlgHost(psf, psfb, img, );
        time(&end);

        std::cout << "Time taken for RL algorithm: " << difftime(end, start) << " seconds" << std::endl;
        
        // call IO function to write the result to disk
        std::cout << "Writing result to disk" << std::endl;
        writeImage(result_3d, "result_3d.raw");
        std::cout << "Result written to disk" << std::endl;
        std::cout << "Done" << std::endl;
    }
}