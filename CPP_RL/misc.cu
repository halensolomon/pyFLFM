#include <array>
#include <complex>
#include <iostream>
#include <vector>
#include <complex>
//#include <thrust/device_vector.h>
#include "thrust/transforms.h"
#include "cufft.h"
#include "cufft_utils.h"

using namespace std;

/// Function to calculate the nearest 2^n that is greater than or equal to 2 * imgSize
__global__ void twoN(int *n, const int imgSize)
{
    /// Caculate nearest 2^n that is greater than or equal to 2 * imgSize
    *n = 1;
    while (*n < 2 * imgSize)
    {
        *n *= 2;
    }
}

/// Function to pad the matrix with zeros to the nearest power of 2 that is greater than or equal to 2 * imgSize
/// Assumes that the 2^n matrix is already allocated (Global functions can't allocate memory nor return values)
__global__ void padMatrix(const thrust::device_vector<thrust::float> *input, thrust::device_vector<thrust::complex> *output, 
const int *imgSize_x, const int *imgSize_y, const int *n, const int *m)
{
    // Pads matrix with zeros to the nearest power of 2 that is greater than or equal to 2 * imgSize
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < *n && idy < *m)
    {
        if (idx < *imgSize_x && idy < *imgSize_y)
        {
            output[idx * (*m) + idy] = input[idx * (*imgSize_y) + idy];
        }
        else
        {
            output[idx * (*m) + idy] = 0.0f;
        }
    }
}

/// Function to reset the padding to zero if for some reason the padding is not zero
__global__ void setPadZero(thrust::device_vector<thrust::complex> *input, const int *n, const int *m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= *n || idy >= *m)
    {
        input[idx * (*m) + idy] = 0.0f;
    }
}

// Function to drop the imaginary part of the complex matrix
__global__ void dropImag(thrust::device_vector<thrust::complex> *input)
{
    input.imag(0.0f);
}


// __host__ void kernfft(const float *kern, thrust::device_vector<thrust::complex> *result, const int n, const int m)
// {
//     /// Take the 2D FFT
//     cufftHandle plan;
//     cufftPlan2d(&plan, n, m, CUFFT_Z2Z);
//     cufftExecZ2Z(plan, kern, result, CUFFT_FORWARD);
//     cufftDestroy(plan);
// }

// __host__ void fftconv(int batch, int dims, const float *img, const float *kernfft, float *result)
// {
//     /// Take the 2D FFT
//     cufftHandle plan;
//     cufftPlanMany(&plan, 2, dims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batch);
//     cufftExecZ2Z(plan, img, img, CUFFT_FORWARD);
//     cufftDestroy(plan);

//     /// Thurst element wise multiplication (treating array like vectors)
//     thrust::transform(img.begin(), img.end(), kernfft.begin(), result.begin(), thrust::multiplies<cudaDoubleComplex>());

//     /// Take the inverse 2D FFT
//     cufftPlan2d(&plan, n, m, CUFFT_Z2Z);
//     cufftExecC2C(plan, result, result, CUFFT_INVERSE);
//     cufftDestroy(plan);

//     /// Drop the imaginary part of the complex matrix and crop the result to the original size
//     dropImag(result);
// }

__global__ void ogCrop(const thrust::device_vector<thrust::complex> *input, thrust::host_vector<thrust::float> *output, 
const int n, const int m, const int imgSize_x, const int imgSize_y, cosnt int numKern)
{
    /// Input is a complex matrix of n x m x numKern
    /// Output is a real matrix of imgSize_x x imgSize_y x numKern
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < imgSize_x && idy < imgSize_y && idz < numKern)
    {
        output[idx * imgSize_y + idy + idz * imgSize_x * imgSize_y] = static_cast<float>(input[idx * m + idy + idz * n * m].real()); // copy-assingment
    }
}

__global__ void fftNorm(thrust::device_vector<thrust::complex> *input, const int n, const int m, const int numKern)
{
    /// Normalize the FFT of the kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < n && idy < m && idz < numKern)
    {
        input[idx * m + idy + idz * n * m] /= static_cast<float>(n * m);
    }
}

__global__ void fftbNorm(thrust::device_vector<thrust::complex> *input, const int n, const int m, const int numKern)
{
    /// Normalize the FFT of the back kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < n && idy < m && idz < numKern)
    {
        input[idx * m + idy + idz * n * m] /= static_cast<float>(n * m);
    }
}