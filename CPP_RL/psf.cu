#include <cuda.h>

__host__ void psfNorm(int kernx, int kerny, int numKern, thrust::device_vector<> kern, thrust::device_vector<> backkern)
{
    /// kernx and kerny are the dimensions of the kernel
    /// numKern is the number of kernels
    /// kern is the thrust::device_vector<float> of the kernel
    /// backkern is the thrust::device_vector<float> of the backward kernel

    thrust::device_vector<float> kernsum(kernx * kerny); // Will be used to normalize the kernel

    /// Need to make iterator that goes [0,1,2,3,4,... 0,1,2,3,4....]
    thrust::device_vector<int> depthwise_addition_iterator(numKern * kernx * kerny);
    thrust::transform(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(numKern * kernx * kerny), depthwise_addition_iterator.begin(), [] (int i) { return i % (kernx * kerny); });

    // make copy of 
    thrust::device_vector<float> kern_copy(numKern * kernx * kerny);
    thrust::copy(thrust::device, kern.begin(), kern.end(), kern_copy.begin());

    thrust::sort_by_key(thrust::device, depthwise_addition_iterator.begin(), depthwise_addition_iterator.end(), kern_copy.begin()); // Sort the kernel by depth
    thrust::reduce_by_key(thrust::device, dep)
    thrust::transform(thrust::device, .begin(), kernsum.end(), thrust::make_constant_iterator(numKern), kernsum.begin(), thrust::multiply<float>()); // Multiply the sum by the number of kernels
    thrust::transform(thrust::device, kernsum.begin(), kernsum.end(), thrust::make_constant_iterator(1e-6), kernsum.begin(), thrust::add<float>()); // Add a small number to avoid division by zero

    // Normalize the kernel, need to divide each pixel by the sum, but the kernsum is kernx * kerny and kerndevptr is numKern * kernx * kerny
    thrust::transform(thrust::device, kerndevptr, kerndevptr + numKern * kernx * kerny, kernsum.begin(), kerndevptr, thrust::divide<float>());


    // Normalize the forward kernel
    for (int i = 0; i < numKern; i++)
    {
        float forward_sum = thrust::reduce(kerndevptr + i * kernx * kerny, kerndevptr + (i + 1) * kernx * kerny);
        thrust::transform(thrust::device, kerndevptr + i * kernx * kerny, kerndevptr + (i + 1) * kernx * kerny, thrust::make_constant_iterator(1e-6), kerndevptr + i * kernx * kerny, thurst::plus<float>()); // Add a small number to avoid division by zerol number to avoid division by zero
        
        //thrust::device_vector<float> kernvec = thurst::device_vector<float>(kerndevptr + i * kernx * kerny, kerndevptr + (i + 1) * kernx * kerny);
        thurst::transform(thrust::device, kerndevptr + i * kern * kerny, kerndevptr + (i + 1) * kernx * kerny, thrust::make_constant_iterator(forward_sum), kerndevptr + i * kernx * kerny, thurst::divide<float>());

        thrust::transform(thrust::device, backkernvec.begin(), backkernvec.end(), kernsum.begin(), kernsum.begin(), thurst::add<float>());

        // Copy the kernel back to the device
        cudaMemcpy(kerndevptr + i * kernx * kerny, kernvec.data(), kernx * kerny * sizeof(float), cudaMemcpyHostToDevice);
        
        // Don't forget to clear the vector
        kernvec.clear();
        kernvec.shrink_to_fit();
    
        for (int i = 0; i < numKern; i++)
    {
        thrust::device_vector<float> backkern = thurst::device_vector<float>(backkerndevptr + i * kernx * kerny, backkerndevptr + (i + 1) * kernx * kerny);
        thrust::transform(backkern.begin(), backkern.end(), kernsum.begin(), backkern.begin(), thurst::divide<float>());
        cudaMemcpy(backkerndevptr + i * kernx * kerny, backkern.data(), kernx * kerny * sizeof(float), cudaMemcpyHostToDevice);

        backkern.clear();
        backkern.shrink_to_fit();
    }

    }

    // Normalize the kernels
    thrust::transform(thrust::device, kernsum.begin(), kernsum.end(), thrust::make_constant_iterator(numKern), kernsum.begin(), thurst::multiply<float>());
    thrust::transform(thrust::device, kernsum.begin(), kernsum.end(), thrust::make_constant_iterator(1e-6), kernsum.begin(), thurst::add<float>());

    // Clear the kernel sum
    kernsum.clear();
    kernsum.shrink_to_fit();
}

__host__ void fftpsf()
{
    // Create cuFFT plans
    cufftHandle plan_kern;
    cufftPlanMany(&plan_kern, 2, dims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batch);
    cufftExecZ2Z(plan_kern, kernArray, kernArray, CUFFT_FORWARD);
    cufftDestroy(plan_kern);
}