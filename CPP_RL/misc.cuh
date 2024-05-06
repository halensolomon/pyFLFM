#ifndef FFTCONV_HPP // include guard
#define FFTCONV_HPP // include guard

void padMatrix(const thrust::device_vector<thrust::float> *input, thrust::device_vector<thrust::complex> *output, 
const int *imgSize_x, const int *imgSize_y, const int *n, const int *m);

void setPadZero(thrust::device_vector<thrust::complex> *input, const int *n, const int *m);

void dropImag(thrust::device_vector<thrust::complex> *input);

void ogCrop(const thrust::device_vector<thrust::complex> *input, thrust::host_vector<thrust::float> *output, 
const int n, const int m, const int imgSize_x, const int imgSize_y, cosnt int numKern);

#endif // include guard

