#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <torch/torch.h>
#include "cuda_rl.cuh"
#include "fftconv.cu"
#include <opencv2/imgcodecs.hpp>
#include "file_io.cu"

namespace fs = std:filesystem;

typedef float2 Complex;

/// One image, multiple kernels: GPU implementation of the 2D convolution operation
/// Zero-copy for image and kernel data
/// Pinned memory for result data, since there will be multiple iterations

// void flipAdd2One(std::vector<std::vector<float>>& originalArray, std::vector<std::vector<float>>& flippedArray) {
//     /// Flip the array and add the original and flipped arrays to one
//     int numRows = originalArray.size();
//     int numCols = originalArray[0].size();

//     std::vector<std::vector<float>> flippedArray(numRows, std::vector<float>(numCols));

//     for (int i = 0; i < numRows; i++) {
//         for (int j = 0; j < numCols; j++) {
//             float sum = originalArray[i][j] + flippedArray[i][j];
//             originalArray[i][j] /= sum;
//             flippedArray[i][j] /= sum;
//             flippedArray[i][numCols - j - 1] = array[i][j];
//         }
//     }
// }

__global__ void rlAlg(int *imgdata, int *kernPath, int *result, int *imgsize, int *kernsize, int *numkern, int* *radius)
{
    // Assumes that the image and kernel are the same size
    // Get the thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // Get the image size; image should just be 2D
    int imgx = imgsize[0];
    int imgy = imgsize[1];
    int imgidx = idx + idy * imgx;

    // Get the kernel size
    int kernx = kernsize[0];
    int kerny = kernsize[1];
    int kernz = kernsize[2];
    int kernidx = idx + idy * kernx + idz * kernx * kerny;

    // Make sure image size and kernel size are the same
    assert (imgx == kernx);
    assert (imgy == kerny);

    int centerx = kernx / 2;
    int centery = kerny / 2;
    
    if ((idx-centerx) ** 2 + (idy-centery) **2  > radius ** 2)
    {
        result[residx] = 0;
    }

    // Do one iteration of the RL algorithm
    // Project the volume to the image space
    // i.e. convole the image with the forward kernel
    for (int i = 0; i < numkern; i++)
    {
        // Get the kernel data
        int kernidx = idx + idy * kernx + idz * kernx * kerny;
        int kernval = kernPath[kernidx];

        // Convolve the image with the kernel
        for (int j = 0; j < kernx; j++)
        {
            for (int k = 0; k < kerny; k++)
            {
                // Update the result for each kernel
                result[idx] += fftconv(imgdata, kernPath[j], result, imgx, imgy)
            }
        }

        // Divide the image elementwise by the result
        thurst::device_vector<int> imgdata_d(imgdata, imgdata + imgx * imgy);
        thurst::device_vector<int> result_d(result, result + imgx * imgy);
        thurst::transform(imgdata_d.begin(), imgdata_d.end(), result_d.begin(), imgdata_d.begin(), thurst::divides<float>());

        // Convolve the image with the backward kernel
        for (int z =0; z < kernz; z++)
        {
            
        }
        
    }
    
}


int main(int argc, char** argv)
{
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Number of GPUs: " << numGPUs << std::endl;

    float *imgData = new float[imgX * imgY];
    float *kernData = new float[imgX * imgY * kernZ];

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

    fileSearch(imgPath, ".tif", imgPaths);
    int numImages = imgPaths.size();

    fileSearch(kernPath, ".tif", kernPaths);
    int numKernels = kernPaths.size();
    
    std::vector<std::vector<float>*> kernPtrStore;

    /// Read all the kernels
    for (int i = 0; i < numKernels; i++)
    {
        /// Read kernel
        std::vector<float>* kernPtr = readImage(kernPaths[i]);
        if (kernPtr != nullptr)
        {
            std::cout << "Kernel read successfullly" << std::end1;
            float* kernPinnedMem; // Pinned memory for kernel data
            size_t kernByteSize = kernPtr->size() * sizeof(float); // Size of the kernel in bytes
            cudaError_t error = cudaMallocHost((void**)&kernPinnedMem, kernByteSize); // Allocate pinned memory for kernel data
            kernPtr.push_back(kernPinnedMem); // Store the kernel data in a vector
        }

        if (error != cudaSuccess) 
        {
            std::cerr << "Failed to allocate pinned memory for kernel data" << std::endl;
        } 
        else 
        {
            cuMemcpy(kernPinnedMem, kernPtr->data(), kernByteSize, cudaMemcpyHostToDevice);
            kernPtrStore.push_back(kernPinnedMem); // Store the allocated pinned memory address
        }
    }

    // Make the backward kernel
    for (int i = 0; i < numKernels, i++)
    {
        std::vector<float>* backKernPtr = kernPtrStore[i];

        float* backKernPinnedMem = backKernPtr->data();
        size_t backKernByteSize = backKernPtr->size() * sizeof(float);

        cudaError_t error = cudaMemcpy(backKernPinnedMem, backKernPtr->data(), backKernByteSize, cudaMemcpyHostToDevice);

        if (error != cudaSuccess)
        {
            std::cerr << "Failed to copy kernel data to device" << std::endl;
        }

        // Flip the kernel
        for (int j = 0; j < backKernPtr->size(); j++)
        {
            backKernPtr[j] = backKernPtr[backKernPtr->size() - j - 1];
        }

        // Normalize the kernel by going through the kernel elementwise
        // Use thrust for this, since all kernels are the same size
        // Add all the kernels elementwise
        thurst::device_vector<float> backKernVec(backKernPtr->data(), backKernPtr->data() + backKernPtr->size());
        thurst::device_vector<float> backKernSum(backKernPtr->size(), 0);
        thurst::transform(backKernVec.begin(), backKernVec.end(), backKernSum.begin(), backKernSum.begin(), thurst::plus<float>());

        // Normalize the kernel
        thurst::transform(backKernVec.begin(), backKernVec.end(), backKernSum.begin(), backKernVec.begin(), thurst::divides<float>());

        // Copy the kernel back to the device
        error = cudaMemcpy(backKernPinnedMem, backKernPtr->data(), backKernByteSize, cudaMemcpyHostToDevice);

        backKernVec.clear();
        backKernSum.clear();

        backKernVec.shrink_to_fit();
        backKernSum.shrink_to_fit();
    }


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

    // Else, continue with the algorithm
    // Allocate memory on the device
    float* results // Result data
    cudaError_t error = cudaMallocHost((void**)&results, kernByteSize); // Allocate pinned memory for kernel data

    if (error != cudaSuccess) 
    {
        std::cerr << "Failed to allocate pinned memory for result data" << std::endl;
    }

    
    




    // Initialize the kernel

    // Initialize the result

    // Initialize the image size

    // Initialize the kernel size

    // Initialize the number of kernels

    // Allocate memory on the device

    // Copy the data to the device



}