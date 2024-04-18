#include <cuda.h>

__host__ void psfNorm()
{
    thrust::device_vector<float> kernsum(kernx * kerny); // Will be used to normalize the kernel

    // Normalize the forward kernel
    for (int i = 0; i < numKernels; i++)
    {
        float forward_sum = thrust::reduce(kerndevptr + i * kernx * kerny, kerndevptr + (i + 1) * kernx * kerny);
        forward_sum += 1e-6; // Add a small number to avoid division by zero
        thrust::device_vector<float> kernvec = thurst::device_vector<float>(kerndevptr + i * kernx * kerny, kerndevptr + (i + 1) * kernx * kerny);
        thurst::transform(thrust::device,kernvec.begin(), kernvec.end(), thrust::make_constant_iterator(forward_sum), kernvec.begin(), thurst::divide<float>());

        thrust::transform(thrust::device, backkernvec.begin(), backkernvec.end(), kernsum.begin(), kernsum.begin(), thurst::add<float>());

        // Copy the kernel back to the device
        cudaMemcpy(kerndevptr + i * kernx * kerny, kernvec.data(), kernx * kerny * sizeof(float), cudaMemcpyHostToDevice);
        
        // Don't forget to clear the vector
        kernvec.clear();
        kernvec.shrink_to_fit();
    }

    // Normalize the kernels
    thrust::transform(thrust::device, kernsum.begin(), kernsum.end(), thrust::make_constant_iterator(numKernels), kernsum.begin(), thurst::multiply<float>());
    thrust::transform(thrust::device, kernsum.begin(), kernsum.end(), thrust::make_constant_iterator(1e-6), kernsum.begin(), thurst::add<float>());

    // Clear the kernel sum
    kernsum.clear();
    kernsum.shrink_to_fit();
}

__host__ void psfbNorm()
{
    // Make the backward kernel
    for (int i = 0; i < numKernels; i++)
    {
        thrust::device_vector<float> backkern = thurst::device_vector<float>(backkerndevptr + i * kernx * kerny, backkerndevptr + (i + 1) * kernx * kerny);
        thrust::transform(backkern.begin(), backkern.end(), kernsum.begin(), backkern.begin(), thurst::divide<float>());
        cudaMemcpy(backkerndevptr + i * kernx * kerny, backkern.data(), kernx * kerny * sizeof(float), cudaMemcpyHostToDevice);

        backkern.clear();
        backkern.shrink_to_fit();
    }

}

__host__ void fftpsf()
{
    // Create cuFFT plans
    cufftHandle plan_kern;
    cufftPlanMany(&plan_kern, 2, dims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batch);
    cufftExecZ2Z(plan_kern, kernArray, kernArray, CUFFT_FORWARD);
    cufftDestroy(plan_kern);
}