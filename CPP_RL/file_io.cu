#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;
using namespace cv;
using namespace fs;
using namespace std;

void fileSearch(const std::string &path, const std::string &ext, std::vector<std::string> &filePaths)
{
    for (const auto &p : fs::recursive_directory_iterator(path))
    {
        if (p.path().extension() == ext)
            filePaths.push_back(p.path().string());
    }
}

std::vector<float>* readImage(const std::string &path)
{
    /// Read the image and store it in a vector
    cv::Mat MatImage = cv::imread(path, cv::IMREAD_UNCHANGED);
    std::vector<float>* ImagePtr = new std::vector<float>;

    if (MatImage.empty())
    {
        std::cout << "Could not read the image: " << path << std::endl;
        exit(0);
    }

    if (MatImage.isContinuous())
    {
        ImagePtr->assign(reinterpret_cast<float*>(MatImage.data), reinterpret_cast<float*>(MatImage.data) + MatImage.total() * MatImage.channels());
        MatImage.release();
    }
    else
    {
        std::cout << "Image is not continuous" << std::endl;
        exit(1);
    }

    return ImagePtr;
}