#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;
using namespace cv;
using namespace fs;
using namespace std;

std::vector<std::string> fileSearch(const std::string &path, const std::string &ext)
{
    std::vector<std::string> filePaths;
    for (const auto &p : fs::recursive_directory_iterator(path))
    {
        if (p.path().extension() == ext)
            filePaths.push_back(p.path().string());
    }
}

struct ImageData 
{
    std::vector<float> *image;
    int width;
    int height;
};

ImageData readImage(const std::string &path)
{
    ImageData img; // Image data struct

    /// Read the image and store it in a vector
    cv::Mat MatImage = cv::imread(path, cv::IMREAD_UNCHANGED);
    std::vector<float>* ImagePtr = new std::vector<float>;

    if (MatImage.empty())
    {
        throw std::runtime_error("Could not read image: " + path)
    }

    if (!MatImage.isContinuous())
    {
        throw std::runtime_error("Image is not continuous: " + path);
    }

    img.data.assign(reinterpret_cast<float*>(MatImage.data), reinterpret_cast<float*>(MatImage.data) + MatImage.total() * MatImage.channels());
    img.width = MatImage.cols;
    img.height = MatImage.rows;
    MatImage.release();

    return img;
}

__host__ void writeImage(std::string &path, thrust::device_vector<thrust::complex> &img, int imgx, int imgy)
{
    std::vector<float> imgVec(img.size());
    thrust::copy(img.begin(), img.end(), imgVec.begin());

    cv::Mat imgMat(imgy, imgx, CV_32FC1, imgVec.data());
    cv::imwrite(path, imgMat);
}