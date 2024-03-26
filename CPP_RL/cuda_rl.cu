#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <torch/torch.h>
#include "cuda_rl.cuh"
#include "fftconv.cu"
#include <opencv2/imgcodecs.hpp>

namespace fs = std:filesystem;

/// One image, multiple kernels: GPU implementation of the 2D convolution operation
/// Zero-copy for image and kernel data
/// Pinned memory for result data, since there will be multiple iterations

void fileSearch(const std::string &path, const std::string &ext std::vector<std::string> &filepaths)
{
    for (const auto &p : fs::recursive_directory_iterator(path))
    {
        if (p.path().extension() == ext)
            filePaths.push_back(p.path().string());
    }
}

std::vector<float>* readImage(const std::string &path)
{
    /// Read the image and store it in a vector
    cv::Mat MatImage = cv::imread(path, cv::IMREAD_UNCHANGED);
    std::vector<float>* ImagePtr = new std::vector<float>;

    if (MatImage.empty())
    {
        std::cout << "Could not read the image: " << path << std::endl;
        exit(0);
    }

    if (MatImage.isContinuous())
    {
        ImagePtr->assign(reinterpret_cast<float*>(MatImage.data), reinterpret_cast<float*>(MatImage.data) + MatImage.total() * MatImage.channels());
        MatImage.release();
    }
    else
    {
        std::cout << "Image is not continuous" << std::endl;
        exit(1);
    }

    return ImagePtr;
}

void flipAdd2One(std::vector<std::vector<float>>& originalArray, std::vector<std::vector<float>>& flippedArray) {
    /// Flip the array and add the original and flipped arrays to one
    int numRows = originalArray.size();
    int numCols = originalArray[0].size();

    std::vector<std::vector<float>> flippedArray(numRows, std::vector<float>(numCols));

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            float sum = originalArray[i][j] + flippedArray[i][j];
            originalArray[i][j] /= sum;
            flippedArray[i][j] /= sum;
            flippedArray[i][numCols - j - 1] = array[i][j];
        }
    }
}

__global__ void rlAlg(int *imgdata, int *kern, int *result, int *imgsize, int *kernsize, int *numkern, int *iterations)
{
    // Get the thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // Get the image size
    int imgx = imgsize[0];
    int imgy = imgsize[1];
    int imgz = imgsize[2];

    // Get the kernel size
    int kernx = kernsize[0];
    int kerny = kernsize[1];
    int kernz = kernsize[2];

    // Get the result size
    int resx = imgx - kernx + 1;
    int resy = imgy - kerny + 1;
    int resz = imgz - kernz + 1;

    // Intialize result data to be a cylinder of ones
    results = 1;

    int centerx = resx / 2;
    int centery = resy / 2;

    for (int kz = 0; kz <= interations; kz++)
    {
        




        
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

    for (int i = 0; i < numKernels, i++)
    {
        std::vector<float>* backKernPtr = kernPtrStore[i];

        float* backKernPinnedMem = backKernPtr->data();
        size_t backKernByteSize = backKernPtr->size() * sizeof(float);

        cudaError_t error = cudaMemcpy(backKernPinnedMem, backKernPtr->data(), backKernByteSize, cudaMemcpyHostToDevice);

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
            cudaError_t error = cudaMallocHost((void**)&imgPinnedMem, imgByteSize); // Allocate pinned memory for image data

            // Copy data from the vector to pinned memory
            std::memcpy(imgPinnedMem, imgPtr->data(), imgByteSize); // Copy the image data to pinned memory

            if (error != cudaSuccess) 
            {
                std::cerr << "Failed to allocate pinned memory for image data" << std::endl;
                delete ImagePtr; // Release ImagePtr if allocation fails
            }

        std::cout << "Image size is: " << imgPtr->size() << std::endl;
        std::cout << "Kernel size is: " << kernPtr->size() << std::endl;

        if (imgPtr->size() != kernPtr->size())
        {
            std::cerr << "Image and kernel sizes do not match" << std::endl;
            exit(2);
        }

        //

        }
    } 



    

    // Initialize the kernel

    // Initialize the result

    // Initialize the image size

    // Initialize the kernel size

    // Initialize the number of kernels

    // Allocate memory on the device

    // Copy the data to the device



}