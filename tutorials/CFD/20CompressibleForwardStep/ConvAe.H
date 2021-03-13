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
struct Encoder : torch::nn::Module {
    explicit Encoder(int64_t latent_dim);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Sequential layer1{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 8, 5).stride(2).padding(0)),
        torch::nn::LeakyReLU(),
        // torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer2{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 5).stride(2).padding(2)),
        torch::nn::LeakyReLU(),
        // torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer3{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 5).stride(2).padding(2)),
        torch::nn::LeakyReLU(),
        // torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer4{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5).stride(2).padding(2)),
        torch::nn::LeakyReLU(),
        // torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer5{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(2)),
        // torch::nn::BatchNorm2d(128),
        torch::nn::LeakyReLU(),
        // torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Linear fc;
};

Encoder::Encoder(int64_t latent_dim)
    : fc(3 * 3 * 128, latent_dim) {
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("layer5", layer5);
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
    x = layer5->forward(x);
    // std::cout << "layer5" << std::endl;
    x = x.view({-1, 3 * 3 * 128});
    return fc->forward(x);
}

/************************************ Decoder **************************/
struct Decoder : torch::nn::Module {
    explicit Decoder(int64_t latent_dim);
    torch::Tensor forward(torch::Tensor x);

    // torch::nn::Sequential layer0{
    //     torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,2})).mode(torch::kNearest)),
    //     torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 4, 5).stride(1).padding(2))
    // };

    // torch::nn::Sequential layer3{
    //     torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,2})).mode(torch::kNearest)),
    //     torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 16, 5).stride(1).padding(2)),
    //     // torch::nn::BatchNorm2d(16),
    //     torch::nn::ReLU()
    // };

    // torch::nn::Sequential layer2{
    //     torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,2})).mode(torch::kNearest)),
    //     torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 5).stride(1).padding(2)),
    //     // torch::nn::BatchNorm2d(32),
    //     torch::nn::ReLU()
    // };

    // torch::nn::Sequential layer1{
    //     torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,2})).mode(torch::kNearest)),
    //     torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 64, 5).stride(1).padding(2)),
    //     // torch::nn::BatchNorm2d(64),
    //     torch::nn::ReLU()
    // };

    torch::nn::Sequential layer1{
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 2).stride(2).padding(1)),
        // torch::nn::BatchNorm2d(64),
        torch::nn::LeakyReLU()
    };

    torch::nn::Sequential layer2{
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, 5).stride(2).padding(2)),
        // torch::nn::BatchNorm2d(64),
        torch::nn::LeakyReLU()
    };

    torch::nn::Sequential layer3{
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(32, 16, 4).stride(2).padding(1)),
        // torch::nn::BatchNorm2d(64),
        torch::nn::LeakyReLU()
    };

    torch::nn::Sequential layer4{
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(16, 8, 4).stride(2).padding(0)),
        // torch::nn::BatchNorm2d(64),
        torch::nn::LeakyReLU()
    };

    torch::nn::Sequential layer5{
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(8, 4, 6).stride(2).padding(0)),
        // torch::nn::BatchNorm2d(64),
        torch::nn::LeakyReLU()
    };

    torch::nn::Linear fc;
};

Decoder::Decoder(int64_t latent_dim)
    : fc(latent_dim, 3 * 3 * 128) {
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("layer5", layer5);
    register_module("fc", fc);
}

torch::Tensor Decoder::forward(torch::Tensor x) {
    // std::cout << "before linear decoder" << x.sizes() << std::endl;
    x = fc->forward(x);
    x = torch::nn::functional::relu(x);
    // std::cout << "after linear decoder" << x.sizes() << std::endl;
    x = x.view({-1, 128, 3, 3});
    // std::cout << "after reshape " << x.sizes() << std::endl;
    x = layer1->forward(x);
    // std::cout << "layer1" << std::endl;
    x = layer2->forward(x);
    // std::cout << "layer2" << std::endl;
    x = layer3->forward(x);
    // std::cout << "layer3" << std::endl;
    x = layer4->forward(x);
    // std::cout << "layer4" << std::endl;
    x = layer5->forward(x);
    return x;
}

/****************************************** Autoencoder ********************************/
struct Autoencoder : torch::nn::Module {
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