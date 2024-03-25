#include <array>
#include <complex>
#include <iostream>
#include <vector>
#include "cufft.h"
#include "cufft_utils.h"

    /// Pad matrix with zeros, then take the 2D FFT
    /// Use cuBLAS.dgmm to perform element-wise multiplication of two matrices
    /// Then take the inverse 2D FFT
    /// Then crop the result to the original size

__device__ void twoN(int *n, int imgSize)
{
    /// Caculate nearest 2^n that is greater than or equal to 2 * imgSize
    *n = 1;
    while (*n < 2 * imgSize)
    {
        *n *= 2;
    }
    }

__device__ void padMatrix(float *d_A, float *d_B, float *h_A, float *h_B, int imgSize)
{
    // Pads matrix with zeros to the nearest power of 2 that is greater than or equal to 2 * imgSize
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < n)
    {
        if (idx < imgSize && idy < imgSize)
        {
            h_A[idx * 2 * imgSize + idy] = d_A[idx * imgSize + idy];
            h_B[idx * 2 * imgSize + idy] = d_B[idx * imgSize + idy];
        }
        else
        {
            h_A[idx * 2 * imgSize + idy] = 0.0f;
            h_B[idx * 2 * imgSize + idy] = 0.0f;
        }
    }
}

__device__ void c2rCropMatrix(float *d_result, float *h_result, int imgSize, int n)
{
    /// Drop the imaginary part of the complex matrix and crop the result to the original size
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < imgSize && idy < imgSize)
    {
        h_result[idx * imgSize + idy] = d_result[idx * n + idy].x;
    }
}

__global__ void fftconv(float *d_A, float *d_B, float *d_result, float *h_result, int imgSize)
{
    /// Find correct size for padding
    int n;
    twoN(&n, imgSize);

    /// Allocate Device Memory for Image and Kernel
    cuComplex *h_A, *h_B;
    cudaMalloc(&h_A, n * n * sizeof(float));
    cudaMalloc(&h_B, n * n * sizeof(float));

    /// Pad matrix with zeros to the nearest power of 2 that is greater than or equal to 2 * imgSize
    padMatrix(d_A, d_B, h_A, h_B, imgSize, n);

    /// Take the 2D FFT
    cufftHandle plan;
    cufftPlan2d(&plan, n, n, CUFFT_R2C);
    cufftExecR2C(plan, h_A, h_A, CUFFT_FORWARD);
    cufftExecR2C(plan, h_B, h_B, CUFFT_FORWARD);
    cufftDestroy(plan);

    /// Use cuBLAS.dgmm to perform element-wise multiplication of two matrices
    cuComplex *d_result;
    cudaMalloc(&d_result, n * n * sizeof(float));

    cublasHandle_t handle; /// cuBLAS handle
    cublasCreate(&handle); /// Create cuBLAS handle
    cublasSdgmm(handle, CUBLAS_SIDE_LEFT, n, n, h_A, n, h_B, n, d_result, n); /// Perform element-wise multiplication
    cublasDestroy(handle); /// Destroy cuBLAS handle

    cudaFree(h_A); /// Free memory on the device
    cudaFree(h_B); /// Free memory on the device

    /// Take the inverse 2D FFT
    cufftPlan2d(&plan, n, n, CUFFT_C2C);
    cufftExecC2C(plan, d_result, h_result, CUFFT_INVERSE);
    cufftDestroy(plan);

    cudaFree(d_result);
}