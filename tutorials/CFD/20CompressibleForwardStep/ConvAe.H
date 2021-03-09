#pragma once

#include <torch/torch.h>

/****************************************** Dataloader ********************************/
class SnapDataset : public torch::data::Dataset<SnapDataset, torch::data::TensorExample>
{
    private:
        torch::Tensor snapshots;

    public:
        explicit SnapDataset(torch::Tensor& snapshots)
            : snapshots{snapshots}{};

        torch::data::TensorExample get(size_t index) override;
        torch::optional<size_t> size() const override;
};

torch::data::TensorExample SnapDataset::get(size_t index)
{
    return snapshots.slice(0, index, index+1);
}

torch::optional<size_t> SnapDataset::size() const
{
    return snapshots.size(0);
};

/****************************************** Encoder ********************************/
class Encoder : public torch::nn::Module {
 public:
    explicit Encoder(int64_t latent_dim);
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Sequential layer1{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 16, 5).stride(1).padding(2)),
        // torch::nn::BatchNorm2d(16),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer2{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 5).stride(1).padding(2)),
        // torch::nn::BatchNorm2d(32),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer3{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5).stride(1).padding(2)),
        // torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer4{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 5).stride(1).padding(2)),
        // torch::nn::BatchNorm2d(128),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Linear fc;
};

Encoder::Encoder(int64_t latent_dim)
    : fc(4 * 4 * 128, latent_dim) {
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("fc", fc);
}

torch::Tensor Encoder::forward(torch::Tensor x) {
    x = layer1->forward(x);
    // std::cout << "layer1" << std::endl;
    x = layer2->forward(x);
    // std::cout << "layer2" << std::endl;
    x = layer3->forward(x);
    // std::cout << "layer3" << std::endl;
    x = layer4->forward(x);
    // std::cout << "layer4" << std::endl;
    x = x.view({-1, 4 * 4 * 128});
    return fc->forward(x);
}

/****************************************** Decoder ********************************/
class Decoder : public torch::nn::Module {
 public:
    explicit Decoder(int64_t latent_dim);
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Sequential layer4{
        torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,2})).mode(torch::kNearest)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 4, 5).stride(1).padding(2))
    };

    torch::nn::Sequential layer3{
        torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,2})).mode(torch::kNearest)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 16, 5).stride(1).padding(2)),
        // torch::nn::BatchNorm2d(16),
        torch::nn::ReLU()
    };

    torch::nn::Sequential layer2{
        torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,2})).mode(torch::kNearest)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 5).stride(1).padding(2)),
        // torch::nn::BatchNorm2d(32),
        torch::nn::ReLU()
    };

    torch::nn::Sequential layer1{
        torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,2})).mode(torch::kNearest)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 64, 5).stride(1).padding(2)),
        // torch::nn::BatchNorm2d(64),
        torch::nn::ReLU()
    };

    torch::nn::Linear fc;
};

Decoder::Decoder(int64_t latent_dim)
    : fc(latent_dim, 4 * 4 * 128) {
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("fc", fc);
}

torch::Tensor Decoder::forward(torch::Tensor x) {
    // std::cout << "before linear decoder" << x.sizes() << std::endl;
    x = fc->forward(x);
    x = torch::nn::functional::relu(x);
    // std::cout << "after linear decoder" << x.sizes() << std::endl;
    x = x.view({-1, 128, 4, 4});
    // std::cout << "after reshape " << x.sizes() << std::endl;
    x = layer1->forward(x);
    // std::cout << "layer1" << std::endl;
    x = layer2->forward(x);
    // std::cout << "layer2" << std::endl;
    x = layer3->forward(x);
    // std::cout << "layer3" << std::endl;
    x = layer4->forward(x);
    // std::cout << "layer4" << std::endl;
    return x;
}

/****************************************** Autoencoder ********************************/
class Autoencoder : public torch::nn::Module {
 public:
    Autoencoder(int64_t latent_dim);

    torch::Tensor forward(torch::Tensor x);

    Decoder decoder;
    Encoder encoder;

    torch::Device device = torch::kCPU;
};

Autoencoder::Autoencoder(int64_t latent_dim)
    : encoder(latent_dim),
      decoder(latent_dim)
{
    register_module("encoder", std::make_shared<Encoder>(encoder));
    register_module("decoder", std::make_shared<Decoder>(decoder));
}

torch::Tensor Autoencoder::forward(torch::Tensor x)
{
    // std::cout << "x: " << x.sizes() << std::endl;
    torch::Tensor z = encoder.forward(x);
    // std::cout << "z: " << z.sizes() << std::endl;
    torch::Tensor x_rec = decoder.forward(z);
    // std::cout << "x rec: " << x_rec.sizes() << std::endl;
    return x_rec;
}