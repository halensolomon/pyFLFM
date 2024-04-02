#include <array>
#include <complex>
#include <iostream>
#include <vector>
#include <complex>
//#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "cufft.h"
#include "cufft_utils.h"

using namespace std;


using namespace std;


/// Pad matrix with zeros, then take the 2D FFT
/// Use cuBLAS.dgmm to perform element-wise multiplication of two matrices
/// Then take the inverse 2D FFT
/// Then crop the result to the original size
/// Pad matrix with zeros, then take the 2D FFT
/// Use cuBLAS.dgmm to perform element-wise multiplication of two matrices
/// Then take the inverse 2D FFT
/// Then crop the result to the original size

__device__ void twoN(int *n, const int imgSize)
__device__ void twoN(int *n, const int imgSize)
{
    /// Caculate nearest 2^n that is greater than or equal to 2 * imgSize
    *n = 1;
    while (*n < 2 * imgSize)
    {
        *n *= 2;
    }
}
}

__device__ void padMatrix(const float *d_A, cudaDoubleComplex *h_A, const int *imgSize_x, const int *imgSize_y, const int *n, const int *m)
__device__ void padMatrix(const float *d_A, cudaDoubleComplex *h_A, const int *imgSize_x, const int *imgSize_y, const int *n, const int *m)
{
    // Pads matrix with zeros to the nearest power of 2 that is greater than or equal to 2 * imgSize
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < *n && idy < *m)
    if (idx < *n && idy < *m)
    {
        if (idx < *imgSize_x && idy < *imgSize_y)
        if (idx < *imgSize_x && idy < *imgSize_y)
        {
            h_A[idx * (*m) + idy] = d_A[idx * (*imgSize_y) + idy];
            h_A[idx * (*m) + idy] = d_A[idx * (*imgSize_y) + idy];
        }
        else
        {
            h_A[idx * (*m) + idy] = 0.0f;
            h_A[idx * (*m) + idy] = 0.0f;
        }
    }
}

__device__ void c2rCropMatrix(const cudaDoubleComplex *input, float *output, const int n, const int m, const int imgSize_x, const int imgSize_y)
{
    /// Drop the imaginary part of the complex matrix and crop the result to the original size
    /// Output is not the same size as the input; it is the same size as the original image (imgSize_x * imgSize_y)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < imgSize_x && idy < imgSize_y)
    {
        output[idx * imgSize_y + idy] = static_cast<float>(input[idx * m + idy].x); /// Drop the imaginary part and crop the result
    }
}

__device__ void cr2Matrix(const cudaDoubleComplex *input, const int n, const int m)
{
    /// Drop the imaginary part of the complex matrix and keep padding

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < m)
    {
        input[idx * m + idy] = static_cast<float>(input[idx * m + idy].x);
    }

    if (idx > imgSize_x && idy > imgSize_y)
    {
        input[idx * m + idy] = 0.0f;
    }
}

__global__ void fftconv(const float *d_A, const float *d_B, float *d_result, int imgSize_x, int imgSize_y)
{

    /// Assumes that the image and kernel are the same size


    /// Assumes that the image and kernel are the same size

    /// Find correct size for padding
    int n;
    twoN(&n, imgSize_x);

    /// Find the correct size for padding
    int m;
    twoN(&m, imgSize_y);
    twoN(&n, imgSize_x);

    /// Find the correct size for padding
    int m;
    twoN(&m, imgSize_y);

    /// Allocate Device Memory for Image and Kernel
    /// h_A and h_B are used to store the padded matrix
    /// h_A and h_B are pointers to the device memory

    cudaDoubleComplex *h_A, *h_B, *h_result;
    cudaMalloc(&h_A, n * m * sizeof(cudaDoubleComplex));
    cudaMalloc(&h_B, n * m * sizeof(cudaDoubleComplex));
    cudaMalloc(&h_result, n * m * sizeof(cudaDoubleComplex));
    /// h_A and h_B are used to store the padded matrix
    /// h_A and h_B are pointers to the device memory

    cudaDoubleComplex *h_A, *h_B, *h_result;
    cudaMalloc(&h_A, n * m * sizeof(cudaDoubleComplex));
    cudaMalloc(&h_B, n * m * sizeof(cudaDoubleComplex));
    cudaMalloc(&h_result, n * m * sizeof(cudaDoubleComplex));

    /// Pad matrix with zeros to the nearest power of 2 that is greater than or equal to 2 * imgSize
    padMatrix(d_A, h_A, imgSize_x, imgSize_y, n, m);
    padMatrix(d_B, h_B, imgSize_x, imgSize_y, n, m);
    padMatrix(d_A, h_A, imgSize_x, imgSize_y, n, m);
    padMatrix(d_B, h_B, imgSize_x, imgSize_y, n, m);

    /// Take the 2D FFT
    cufftHandle plan;
    cufftPlan2d(&plan, n, m, CUFFT_C2C);
    cufftExecC2C(plan, h_A, h_A, CUFFT_FORWARD);
    cufftExecC2C(plan, h_B, h_B, CUFFT_FORWARD);
    cufftDestroy(plan);

    /// Thurst element wise multiplication (treating array like vectors)
    thrust::device_ptr<int> thrust_h_A(h_A);
    thrust::device_ptr<int> thrust_h_B(h_B);
    thrust::device_ptr<int> thrust_h_result(h_result);
    thrust::transform(thrust_h_A, thrust_h_A + n * m, thrust_h_B, thrust_h_result, thrust::multiplies<cudaDoubleComplex>());

    cudaFree(h_A); /// Free memory on the device
    cudaFree(h_B); /// Free memory on the device

    /// Take the inverse 2D FFT
    cufftPlan2d(&plan, n, m, CUFFT_C2C);
    cufftExecC2C(plan, h_result, h_result, CUFFT_INVERSE);
    cufftDestroy(plan);

    /// Drop the imaginary part of the complex matrix and crop the result to the original size
    c2rCropMatrix(h_result, d_result, n, m, imgSize_x, imgSize_y);

    /// Free memory on the device
    cudaFree(h_result);
}

__global__ cudaDoubleComplex fftconvKeepPad(const float *d_A, const float *d_B, int imgSize_x, int imgSize_y)
{

    /// Assumes that the image and kernel are the same size

    /// Find correct size for padding
    int n;
    twoN(&n, imgSize_x);

    /// Find the correct size for padding
    int m;
    twoN(&m, imgSize_y);

    /// Allocate Device Memory for Image and Kernel
    /// h_A and h_B are used to store the padded matrix
    /// h_A and h_B are pointers to the device memory

    cudaDoubleComplex *h_A, *h_B, *h_result;
    cudaMalloc(&h_A, n * m * sizeof(cudaDoubleComplex));
    cudaMalloc(&h_B, n * m * sizeof(cudaDoubleComplex));
    cudaMalloc(&h_result, n * m * sizeof(cudaDoubleComplex));

    /// Pad matrix with zeros to the nearest power of 2 that is greater than or equal to 2 * imgSize
    padMatrix(d_A, h_A, imgSize_x, imgSize_y, n, m);
    padMatrix(d_B, h_B, imgSize_x, imgSize_y, n, m);

    /// Take the 2D FFT
    cufftHandle plan;
    cufftPlan2d(&plan, n, m, CUFFT_C2C);
    cufftExecC2C(plan, h_A, h_A, CUFFT_FORWARD);
    cufftExecC2C(plan, h_B, h_B, CUFFT_FORWARD);
    cufftDestroy(plan);

    /// Thurst element wise multiplication (treating array like vectors)
    thrust::device_ptr<cudaDoubleComplex> thrust_h_A(h_A);
    thrust::device_ptr<cudaDoubleComplex> thrust_h_B(h_B);
    thrust::device_ptr<cudaDoubleComplex> thrust_h_result(h_result);
    thrust::transform(thrust_h_A, thrust_h_A + n * m, thrust_h_B, thrust_h_result, thrust::multiplies<cudaDoubleComplex>());

    cudaFree(h_A); /// Free memory on the device
    cudaFree(h_B); /// Free memory on the device

    /// Take the inverse 2D FFT
    cufftPlan2d(&plan, n, m, CUFFT_C2C);
    cufftExecC2C(plan, h_result, h_result, CUFFT_INVERSE);
    cufftPlan2d(&plan, n, m, CUFFT_C2C);
    cufftExecC2C(plan, h_result, h_result, CUFFT_INVERSE);
    cufftDestroy(plan);

    /// Drop the imaginary part of the complex matrix and crop the result to the original size
    c2rmatrix(h_result, n, m);
    
    /// Deal with dangling pointers
    h_A = nullptr;
    h_B = nullptr;

    return h_result;
}