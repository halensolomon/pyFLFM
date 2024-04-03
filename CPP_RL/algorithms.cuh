#ifndef ALGORITHMS
#define ALGORITHMS

__global__ void rlAlg(float *img, float *kernArray, float *backkernArray, float *result_2d, float *result_3d, int *imgsize, int *kernsize, int *numkern, int* *radius)

__device__ void cylMask(float *input, int idx, int idy, int idz, int imgx, int imgy, int centerx, int centery, int radius)

#endif