#include <torch/torch.h>
#include <torch/serialize.h>

namespace pt_load
{
    void load_tensor(std::vector<torch::Tensor> &tensor_vec, std::string filename)
    {
        torch::load(tensor_vec, filename);
    }
}