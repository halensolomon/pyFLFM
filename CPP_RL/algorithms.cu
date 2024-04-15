#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

__device__ void cylMask(float *input, int idx, int idy, int idz, int imgx, int imgy, int centerx, int centery, int radius)
{
    if ((idx - centerx) * (idx - centerx) + (idy - centery) * (idy - centery) > (radius) * (radius))
    {
        input[idx + idy * imgx + idz * imgx * imgy] = 0;
    }
    else 
    {
        input[idx + idy * imgx + idz * imgx * imgy] = 1;
    }
}


__host__ void rlAlgHost()
{
    // Host needs to create enough memory for the image, kernel, and result
    // Host needs to copy the image and kernel data to the device

    // Host needs to create a initial guess for the volume
    thrust::device_vector<int> cyl(imgx * imgy * kernz); // Create a cylinder mask
    thrust::transform(cyl.begin(), cyl.end(), cyl.begin(), 0); // Initialize the cylinder mask to 0

    cylMask<<< >>>( ); // Apply the cylinder mask

    // Host needs to create a temporary variable to store the 2D results
    float* img_2d;
    float* img_2d_sum;
    float* vol_3d;
    
    // Allocate memory for the 2D results
    cudaMalloc((void**)&img_2d, imgx * imgy * kernz * sizeof(float));
    cudaMalloc((void**)&img_2d_sum, imgx * imgy * sizeof(float));
    cudaMalloc((void**)&vol_3d, imgx * imgy * kernz * sizeof(float));

    // Could use a for loop, but that would be serializing the process
    for (int i = 0; i < itr; i++)
    {
        rlAlg<<<>>>(img_2d, kernArray, backkernArray, img_2d_sum, vol_3d, imgsize, kernsize, numkern, radius);
    }
}

__global__ void rlAlg(float *img, float *kernArray, float *backkernArray, float *result_2d, float *result_3d, int *imgsize, int *kernsize, int *numkern, int *radius)
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
    int kernidx = idz + idx * kernx + idz * kernx * kerny;

    // Make sure image size and kernel size are the same
    assert (imgx == kernx);
    assert (imgy == kerny);

    int centerx = kernx / 2;
    int centery = kerny / 2;

    // Update the result for each kernel
    fftconv(img, kern, temp, imgx, imgy);

    result_2d[idx] = temp_img_res_2d[idx]; // Store the result in the 2D result array

    __synchronize();

    // Divide the image elementwise by the result
    // Temporary variable to store the division result
    thrust::device_vector<float> ratio(imgx * imgy);
    thurst::transform(result_2d.begin(), result_2d.end(), thrust::make_constant_iterator(1e-6), result_2d.begin(), thurst::add<float>()); // Add a small number to avoid division by zero
    thurst::transform(img.begin(), img.end(), result_2d.begin(), imgdata_2d.begin(), thurst::divides<float>());

    backProp<<<>>>();

    // Convolve the image with the backward kernel
    for (int z = 0; z < kernz; z++)
    {
        fftconv(imgdata, backkernArray[z], ratio[&ratio + z * imgx * imgy], imgx, imgy)
    }

    // Multiply the 3D result by the ratio
    //thurst::device_vector<int> result_3d(result_3d, result_3d + kernz * imgx * imgy);

    thurst::transform(result_3d.begin(), result_3d.end(), ratio.begin(), result_3d.begin(), thurst::multiplies<float>()); // Multiply the 3D result by the ratio

    ratio.clear(); // Might be unnecessary since all the variables will be out of scope
    ratio.shrink_to_fit();
};

__device__ backProp()
{
    // Backward propagation
    fftconv(img, backkernArray, temp, imgx, imgy);
};