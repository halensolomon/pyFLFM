#include <iostream>
#include <cuda.h>

#include "opencv2/imgcodecs.hpp"
#include "cuda_rl.cuh"
#include "algorithms.cuh"
#include "fftconv.cuh"
#include "file_io.cuh"

#include "misc.cuh"
#include "algorithms.cuh"
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

    /// Ask the user for the number of iterations
    int itr;
    std::cout << "Enter the number of iterations: ";
    std::cin >> itr;

    str resultPath = "C:/some/path/to/results/";
    std::cout << "Enter path to save the result: ";
    std::cin >> resultPath;
    
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

    /// Two-Nify the kernel
    int nPoT_x = 0; // nearest Power of Two for x
    int nPoT_y = 0; // nearest Power of Two for y
    twoN(&nPoT_x, imgx); // Calculate the nearest power of 2 that is greater than or equal to 2 * imgSize
    twoN(&nPoT_y, imgy); // Calculate the nearest power of 2 that is greater than or equal to 2 * imgSize

    /// Allocate memory for the image and kernel data
    thrust::device_vector<thrust::complex<double>> img_padded(nPoT_x * nPoT_y);
    thrust::device_vector<thrust::complex<double>> kern_padded(nPoT_x * nPoT_y * numKernels);
    thrust::device_vector<thrust::complex<double>> backkern_padded(nPoT_x * nPoT_y * numKernels);

    /// Grab the raw pointers for the device vectors
    thrust::complex<double> *_img_padded = thrust::raw_pointer_cast(img.data());
    thrust::complex<double> *_kern_padded = thrust::raw_pointer_cast(kernArray.data());
    thrust::complex<double> *_backkern_padded = thrust::raw_pointer_cast(backKernArray.data());

    /// Load PSF and PSFb into device memory and normalize
    psfNorm(kerndevptr, kernx, kerny, numKernels);
    psfbNorm(backkerndevptr, kernx, kerny, numKernels);

    /// Copy kernels to the padded vector
    padMatrix(kerndevptr, _kern_padded, kernx, kerny, nPoT_x, nPoT_y);
    padMatrix(backkerndevptr, _backkern_padded, kernx, kerny, nPoT_x, nPoT_y);

    /// Take the FFT of the kernel
    fftpsf(_kern_padded);
    fftpsf(_backkern_padded);

    /// Allocate memory for the image and kernel data
    thrust::device_vector<thrust::complex<double>> img_padded(nPoT_x * nPoT_y);
    thrust::device_vector<thrust::complex<double>> kern_padded(nPoT_x * nPoT_y * numKernels);
    thrust::device_vector<thrust::complex<double>> backkern_padded(nPoT_x * nPoT_y * numKernels);

    /// Grab the raw pointers for the device vectors
    thrust::complex<double> *_img_padded = thrust::raw_pointer_cast(img.data());
    thrust::complex<double> *_kern_padded = thrust::raw_pointer_cast(kernArray.data());
    thrust::complex<double> *_backkern_padded = thrust::raw_pointer_cast(backKernArray.data());

    /// Copy kernels to the padded vector
    padMatrix(kerndevptr, _kern_padded, kernx, kerny, nPoT_x, nPoT_y);
    padMatrix(backkerndevptr, _backkern_padded, kernx, kerny, nPoT_x, nPoT_y);

    /// Orignial kernel data is no longer needed
    cudaFree(kerndevptr);
    cudaFree(backkerndevptr);

    /// Run the RL algorithm
    thrust::host_vector<float>> result_3d(imgx * imgy * numKern);

    /// Read images sequentially
    for (i = 0, i < numImages, i++)
    {
        std::vector<float>* imgPtr = readImage(imgPaths[i]);
        if (imgPtr != nullptr)
        {
            std::cout << "Image read successfully" << std::endl;
            cudaError_t error = cudaHostAlloc((void**)&imgPinnedMem, imgByteSize); // Allocate pinned memory for image data

            // Copy data from imgPtr to img_padded
            cudaMemcpy(imgdevptr, imgPtr->data(), imgPtr->size(), cudaMemcpyHostToDevice);

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

        {
        time_t start, end;
        time(&start);
        rlAlgHost(psf, psfb, img, );
        time(&end);
        std::cout<< "Image " << i << " processed"
        std::cout << "Time taken for RL algorithm: " << difftime(end, start) << " seconds" << std::endl;
        
        // call IO function to write the result to disk
        std::cout << "Writing result to disk" << std::endl;
        writeImage(result_3d, "result_3d.raw");
        std::cout << "Result written to disk" << std::endl;
        
        }
    std::cout << "Done" << std::endl;

    }

}