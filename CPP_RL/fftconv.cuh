#ifndef FFTCONV_HPP // include guard
#define FFTCONV_HPP // include guard

void padMatrix(float *d_A, float *d_B, float *h_A, float *h_B, int imgSize, int n);
void c2rCropMatrix(float *d_result, float *h_result, int imgSize, int n);
void fftconv(float *d_A, float *d_B, float *d_result, float *h_result, int imgSize);

#endif // include guard

