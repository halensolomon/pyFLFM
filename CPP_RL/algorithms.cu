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
    
    // Declaration of the cylinder mask
    void cylMask(float *input, int idx, int idy, int idz, int imgx, int imgy, int centerx, int centery, int radius)
    {
        if ((idx - centerx) * (idx - centerx) + (idy - centery) * (idy - centery) > (*radius) * (*radius))
        {
            input[idx + idy * imgx + idz * imgx * imgy] = 0;
        }
        else 
        {
            input[idx + idy * imgx + idz * imgx * imgy] = 1;
        }
    }

    thrust::device_vector<int> cyl(imgx * imgy* kernz);
    cylMask(cyl, idx, idy, idz, imgx, imgy, centerx, centery, radius);

    // Apply cylinder mask
    thrust::transform()
    
    // Make sure result_2d only contains 0.0f
    result_2d[imgidx] = 0.0f;

    // Do one iteration of the RL algorithm
    // Project the volume to the image space
    // i.e. convole the image with the forward kernel
    for (int i = 0; i < numkern; i++)
    {
        // Get the kernel data
        int kernidx = idx + idy * kernx + idz * kernx * kerny;
        int kernval = kernArray[kernidx];

        // Grab the specific kernel
        thrust::device_vector<float> kern(kernArray + i * kernx * kerny, kernArray + (i + 1) * kernx * kerny);

        // Make temporary variable to store fftconv result
        thrust::device_vector temp(imgx * imgy);

        // Update the result for each kernel
        fftconv(img, kern, temp, imgx, imgy)

        result_2d[idx] += temp
    }
    // temp should be out of scope here, so it should be automatically deleted

    // Divide the image elementwise by the result
    // Temporary variable to store the division result
    thrust::device_vector<float> ratio(imgx * imgy);
    thurst::transform(result_2d.begin(), result_2d.end(), thrust::make_constant_iterator(1e-6), result_2d.begin(), thurst::add<float>()); // Add a small number to avoid division by zero
    thurst::transform(img.begin(), img.end(), result_2d.begin(), imgdata_2d.begin(), thurst::divides<float>());

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
}