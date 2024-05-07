#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/complex.h>
#include <cuComplex.h>
#include <cufft.h>

#include "misc.cuh"

__global__ void cylMask(float *input, int idx, int idy, int idz, int imgx, int imgy, int centerx, int centery, int radius)
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
///<<<griddim, blockdim, sharedmem, stream>>>

__host__ void rlAlgHost(int itr, int imgx, int imgy, int numKern, 
thrust::device_vector<thrust::complex<double>> img, thrust::device_vector<thrust::complex<double>> kernArray, 
thrust::device_vector<thrust::complex<double>> backKernArray, thrust::host_vector<thrust::complex<double>> result_3d)
{   
    // Arbitrary number of streams
    const int numStreams = 5;
    cudaStream_t streams[numStreams];
    
    for (int i = 0; i < numStreams; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    int nPoT_x = 0; // nearest Power of Two for x
    int nPoT_y = 0; // nearest Power of Two for y
    twoN(&nPoT_x, imgx); // Calculate the nearest power of 2 that is greater than or equal to 2 * imgSize
    twoN(&nPoT_y, imgy); // Calculate the nearest power of 2 that is greater than or equal to 2 * imgSize

    // Host needs to create enough memory for the image, kernel, and result
    // Host needs to copy the image and kernel data to the device
    // Initialize the temporary variables
    thrust::device_vector<thrust::complex<double>> predicted_img(nPoT_x * nPoT_y * numKern);
    thrust::device_vector<thrust::complex<double>> result_sum(nPoT_x * nPoT_y);
    thrust::device_vector<thrust::complex<double>> result_back(nPoT_x * nPoT_y * numKern);
    thrust::device_vector<thrust::complex<double>> predicted_volume(nPoT_x * nPoT_y * numKern);
    thrust::device_vector<thrust::complex<double>> ratio(nPoT_x * nPoT_y);
    
    // Grab the raw pointers for the device vectors
    thrust::complex<double> *_img = thrust::raw_pointer_cast(img.data());
    thrust::complex<double> *_kernArray = thrust::raw_pointer_cast(kernArray.data());
    thrust::complex<double> *_backKernArray = thrust::raw_pointer_cast(backKernArray.data());
    thrust::complex<double> *_predicted_img = thrust::raw_pointer_cast(predicted_img.data());
    thrust::complex<double> *_result_sum = thrust::raw_pointer_cast(result_sum.data());
    thrust::complex<double> *_result_back = thrust::raw_pointer_cast(result_back.data());
    thrust::complex<double> *_predicted_volume = thrust::raw_pointer_cast(predicted_volume.data());
    thrust::complex<double> *_ratio = thrust::raw_pointer_cast(ratio.data());

    // Copy image to the padded vector
    padMatrix(_img, _predicted_img, &imgx, &imgy, &nPoT_x, &nPoT_y);
    
    // Initialize the cylinder mask
    thrust::fill(predicted_volume.begin(), predicted_volume.end(), 0.0f); // Initialize the array to 0
    cylMask<<<    >>>(_predicted_volume); // Apply the cylinder mask

    // Make transform iterator (0,1,2,...,elem-1)
    thrust::device_vector<int> depthwise_addition_iterator(elem);
    thrust::transform(thrust::device<>, thrust::make_counting_iterator(0), thrust::make_counting_iterator(elem), depthwise_addition_iterator.begin(), [] (int i) { return i % img_elem; });

    // Create cuFFT plans
    cufftHandle plan_vol;
    cufftHandle plan_result;

    cufftPlanMany(&plan_vol, 2, dims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batch);
    cufftExecZ2Z(plan_vol, kernArray, kernArray, CUFFT_FORWARD);

    cufftPlan2D(&plan_result, nPoT_x, nPoT_y, CUFFT_Z2Z);

    for (int i = 0; i < itr; i++)
    {
        // Forward FFT the volume
        cufftExecZ2Z(_plan_vol, _predicted_volume, _predicted_volume, CUFFT_FORWARD);
        thrust::transform(thrust::device, kernArray.begin(), kernArray.end(), predicted_volume.begin(), predicted_img.begin(), thrust::multiplies<thrust::complex<double>>()); // Multiply the 3D volume by the kernel
        // Sum the depthwise addition
        thrust::reduce_by_key(depthwise_addition_iterator.begin(), depthwise_addition_iterator.end(), predicted_img.begin(), thrust::make_discard_iterator(), result_sum.begin()); // Sum the depthwise addition
        thrust::transform(thrust::device, result_sum.begin(), result_sum.end(), thrust::make_constant_iterator(1e-6), result_sum.begin(), thrust::plus<thrust::complex<double>>()); // Add a small number to avoid division by zero
        // Divide the image by the sum
        thrust::transform(thrust::device, img.begin(), img.end(), result_sum.begin(), ratio.begin(), thrust::divides<thrust::complex<double>>());
        // Multiply the ratio by the backward kernel in the frequency domain and take the inverse FFT
        thrust::transform(thrust::device, backkernArray.begin(), backkernArray.end(), ratio.begin(), result_back.begin(), thrust::multiplies<thrust::complex<double>>()); // Multiply the ratio by the backward kernel
        cufftExecZ2Z(_plan_result, _result_back, _result_back, CUFFT_BACKWARD); // Backward FFT the result
        cufftExecZ2Z(_plan_result, _predicted_volume, _predicted_volume, CUFFT_Backward); // Inverse FFT the predicted volume
        thrust::transform(thrust::device, predicted_volume.begin(), predicted_volume.end(), result_back.begin(), predicted_volume.begin(), thrust::multiplies<thrust::complex<double>>()); // Multiply the 3D result by the ratio
    }

    /// Destroy the cuFFT plans
    cufftDestroy(plan_vol);
    cufftDestroy(plan_result);

    // Copy the result back to the host device
    ogCrop(predicted_volume, result_3d); // Crop the result to the original size
}

// This requires cuFFTdx instead of cuFFT
// __device__ void rlAlForward(float *img, float *kernArray, float *backkernArray, float *result_2d, 
// float *result_3d, int *imgsize, int *kernsize, int *numkern, int *radius)
// {
//     // Assumes that the image and kernel are the same size
//     // Get the thread index
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int idz = blockIdx.z * blockDim.z + threadIdx.z;

//     // Get the image size; image should just be 2D
//     int imgx = imgsize[0];
//     int imgy = imgsize[1];
//     int imgidx = idx + idy * imgx;

//     // Get the kernel size
//     int kernx = kernsize[0];
//     int kerny = kernsize[1];
//     int kernz = kernsize[2];
//     int kernidx = idz + idx * kernx + idz * kernx * kerny;

//     // Make sure image size and kernel size are the same
//     assert (imgx == kernx);
//     assert (imgy == kerny);

//     int centerx = kernx / 2;
//     int centery = kerny / 2;

//     // Update the result for each kernel
//     fftconv(img, kern, temp, imgx, imgy);

//     result_2d[idx] = temp_img_res_2d[idx]; // Store the result in the 2D result array
// }

// __device__ void rlAlBackward(float *img, float *kernArray, float *backkernArray, float *result_2d, 
// float *result_3d, int *imgsize, int *kernsize, int *numkern, int *radius)
// {
//     fftconv(imgdata, backkernArray[z], ratio[&ratio + z * imgx * imgy], imgx, imgy)
// }

// __device__ void cylMask(float *input, int idx, int idy, int idz, int imgx, int imgy, int centerx, int centery, int radius)
// {
//     if ((idx - centerx) * (idx - centerx) + (idy - centery) * (idy - centery) > (radius) * (radius))
//     {
//         input[idx + idy * imgx + idz * imgx * imgy] = 0;
//     }
//     else 
//     {
//         input[idx + idy * imgx + idz * imgx * imgy] = 1;
//     }
// }
