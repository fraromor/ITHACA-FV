/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
-------------------------------------------------------------------------------

License
    This file is part of ITHACA-FV

    ITHACA-FV is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ITHACA-FV is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

/// \file
/// Source file of the NonlinearReducedBurgers class

#include <torch/script.h>
#include <torch/torch.h>
#include "torch2Eigen.H"
#include "torch2Foam.H"
#include "Foam2Eigen.H"
#include "ITHACAstream.H"
#include <chrono>
#include "cnpy.H"
#include <algorithm>

#include "NonlinearReducedBurgers_PI.H"

using namespace ITHACAtorch;

// * * * * * * * * * * * * * * * Autoencoder * * * * * * * * * * * * * * * * //
Encoder::Encoder(int64_t image_size, int64_t h_dim, int64_t z_dim)
    : fc1(image_size, h_dim),
      fc2(h_dim, z_dim)
{
    register_module("fc1", fc1);
    register_module("fc2", fc2);
}

torch::Tensor Encoder::forward(torch::Tensor x)
{
    // std::cout << "out1enc" << x.size(0) << " " <<  x.size(1) << std::endl;
    torch::Tensor out = torch::nn::functional::relu(fc1(x));
    // std::cout << "out1enc" << out.size(0) <<  " " << out.size(1) << std::endl;
    out = torch::nn::functional::relu(fc2(out));
    return out;
}

Decoder::Decoder(int64_t image_size, int64_t h_dim, int64_t z_dim)
    : fc1(z_dim, h_dim),
      fc2(h_dim, image_size)
{
    register_module("fc1", fc1);
    register_module("fc2", fc2);
}

torch::Tensor Decoder::forward(torch::Tensor z)
{
    torch::Tensor out = torch::nn::functional::relu(fc1(z));
    out = torch::nn::functional::relu(fc2(out));
    return out;
}


AE::AE(int64_t image_size, int64_t h_dim, int64_t z_dim)
    : encoder(image_size, h_dim, z_dim),
      decoder(image_size, h_dim, z_dim)
{
    register_module("encoder", std::make_shared<Encoder>(encoder));
    register_module("decoder", std::make_shared<Decoder>(decoder));
}

torch::Tensor AE::forward(torch::Tensor x)
{
    x = (x - bias_inp) / scale_inp;
    torch::Tensor z = encoder.forward(x);
    torch::Tensor x_rec = decoder.forward(z);
    x_rec = x_rec * scale_inp + bias_inp;
    return x_rec;
}

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

NonlinearReducedBurgers::NonlinearReducedBurgers()
{
    para = ITHACAparameters::getInstance();
}

NonlinearReducedBurgers::NonlinearReducedBurgers(Burgers &FOMproblem, fileName net_path, int dim_latent, Eigen::MatrixXd latent_initial)
    : Nphi_u{dim_latent},
      problem{&FOMproblem}
{

    // problem->L_Umodes is used to create volVectorFields
    embedding = autoPtr<Embedding>(new Embedding(Nphi_u, net_path, FOMproblem, latent_initial));

    // FOMproblem is only needed for initial conditions
    newton_object = newton_nmlspg_burgers(Nphi_u, 2*embedding->output_dim, FOMproblem, embedding.ref(), problem->L_Umodes[0]);
}

Embedding::Embedding(int dim, fileName net_path, Burgers &problem, Eigen::MatrixXd lat_init) : latent_dim{dim}, latent_initial{lat_init}, problem{&problem}
{
    // get the number of degrees of freedom relative to a single component
    output_dim = problem.L_Umodes[0].size(); // 3600

    autoencoder = autoPtr<AE>(new AE(output_dim*2, output_dim*4, latent_dim));
    loadData();
    train();

    // define initial velocity field _U0 used to define the reference snapshot
    // and initialize decoder output variable g0
    _U0 = autoPtr<volVectorField>(new volVectorField(problem.L_Umodes[0]));
    _g0 = autoPtr<volVectorField>(new volVectorField(problem.L_Umodes[0]));

    // declare input of decoder of type IValue since the decoder is loaded from pytorch
    torch::Tensor latent_initial_tensor = torch2Eigen::eigenMatrix2torchTensor(latent_initial);

    std::cout << "LATENT INITIAL" << latent_initial_tensor << std::endl;

    torch::Tensor tensor = autoencoder->decoder.forward(std::move(latent_initial_tensor)).to(torch::kCPU);

    // add the z component to the tensor as a zero {1, 60, 60} tensor and
    // reshape the tensor s.t. the components x,y,z of a single cell center are
    // contiguous in memory (this is necessary for torch2field method)
    auto tensor_stacked = torch::cat({std::move(tensor).reshape({2, 60, 60}), torch::zeros({1, 60, 60})}, 0).reshape({3, -1}).transpose(0, 1).contiguous();


    auto g0 = torch2Foam::torch2Field<vector>(tensor_stacked);
    _g0.ref().ref().field() = std::move(g0);
    // save_field.append(_g0.clone();
    // ITHACAstream::exportFields(save_field, "./REF", "g0");
}

torch::Tensor Embedding::regularizerl1(torch::Device& device, float factor=0.001)
{
    torch::Tensor loss_reg = torch::zeros({1});
    loss_reg = loss_reg.to(device);

    for (auto& param: autoencoder->parameters())
    {
        loss_reg += torch::norm(param, 1);
    }

    loss_reg *= factor;
    return loss_reg;

}

// void Embedding::adjacency_mask(int shift)
// {
//     std::vector<torch::Tensor> active_params;
//     for (auto & param: autoencoder->named_parameters().keys())
//     {
//         std::cout << "param " << param << std::endl;
//     }
//     auto W1 = autoencoder->named_parameters()["encoder.fc1.weight"];
//     auto W2 = autoencoder->named_parameters()["decoder.fc2.weight"];

//     W1 = W1.set_requires_grad(false);
//     auto W1_clone = W1.clone();
//     W1_clone = W1_clone.set_requires_grad(false);
//     std::cout << "clone: " << W1_clone.requires_grad() << W1_clone[0, 0].requires_grad()<< std::endl;

//     W1_clone[0, 0] = W1_clone[0, 0].set_requires_grad(true);
//     std::cout << "clone: " << W1_clone.requires_grad() << W1_clone[0, 0].requires_grad()<< std::endl;

//     for (int j=0; j<W1_clone.size(1); j++)
//     {
//         std::cout << "column: " << j << std::endl;

//         long lower = std::max(0, shift * j - 4 * 60 * 2);
//         long upper = std::min(W1_clone.size(0), long(shift * j + 4 * 60 * 2));

//         std::cout << "length " << upper-lower << std::endl;

//         W1_clone.slice(1, lower, upper) = W1_clone.slice(1, lower, upper).fill_(0).set_requires_grad(true);
//         std::cout << W1_clone[0, shift * j + 4 * 60 * 2+1].requires_grad() << std::endl;
//     }
//     std::cout << "type: " << W1_clone.dtype() << std::endl;
//     W1 = W1 - W1_clone;

//     for (int j=0; j<W1_clone.size(1); j++)
//     {
//         std::cout << W1_clone[0, j].requires_grad() << std::endl;
//     }

//     // Eigen::MatrixXd mat = torch2Eigen::torchTensor2eigenMatrix<float>(W1);
//     // cnpy::save(mat, "spy.npy");
//     // std::cout << "saved" << std::endl;


//     // for (int j=0; j<W1.size(1); j++)
//     // {
//     //     std::cout << "column: " << j << std::endl;

//     //     // int lower = std::max(0, shift * j - 4 * 60 * 2);
//     //     // int upper = std::min(W1.size(0), shift * j + 4 * 60 * 2);

//     //     long lower_below = shift * j + 4 * 60 * 2;
//     //     long upper_above = std::min(W1.size(0), long(shift * j - 4 * 60 * 2));

//     //     // * below shifted diagonal element
//     //     if (upper_above > 0)
//     //     {
//     //         W1.slice(1, 0, upper_above) = W1.slice(1, 0, upper_above).fill_(0).set_requires_grad(false);
//     //     }

//     //     // * aboveshifted diagonal element
//     //     if (lower_below < W1.size(0))
//     //     {
//     //         W1.slice(1, lower_below, W1.size(0)) = W1.slice(1, lower_below, W1.size(0)).fill_(0).set_requires_grad(false);
//     //     }

//     // }



//     // for (auto& param: )
//     // {
//     //     param = param.requires_grad(false);
//     // }

//     // std::cout << autoencoder->parameters();

// }

void Embedding::train()
{
    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(torch::kCPU);//cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t batch_size = 100;
    const size_t num_epochs = 100;
    const double learning_rate = 1e-2;

    // Autoencoder training parameters
    autoencoder->to(device);
    autoencoder->bias_inp = autoencoder->bias_inp.to(device);
    autoencoder->scale_inp = autoencoder->scale_inp.to(device);
    // adjacency_mask(2);

    optimizer = new torch::optim::Adam(autoencoder->parameters(),
                                               torch::optim::AdamOptions(learning_rate));
    // Generate a data loader.
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    std::move(data_set()),
    batch_size);

    // load test dataset
    auto ptrList = problem->Ufield;
    int Nrows = ptrList.size();
    int Ncols = ptrList[0].size() * 2;
    torch::Tensor train_snap_unscaled = torch::randn({Nrows, Ncols});

    std::cout << "Loading dataset test ... " << Nrows << "x" << Ncols << std::endl;

    for (auto i = 0; i < ptrList.size(); i++)
    {
        auto imported = torch2Foam::field2Torch(ptrList[i]).squeeze();
        // std::cout << imported.size(0) << std::endl;
        train_snap_unscaled.slice(0, i, i + 1) = imported.reshape({output_dim, 3}).slice(1, 0, 2).reshape(output_dim*2);
    }

    std::cout << "Finished loading test ... " << train_snap_unscaled.size(0) << std::endl;

    std::cout << "Training...\n";
    auto test_loss_norm = torch::frobenius_norm(torch::abs(train_snap_unscaled));

    autoencoder->train();

    for (int64_t epoch = 1; epoch <= num_epochs; ++epoch)
    {
        torch::Tensor loss{torch::zeros({1})};
        torch::Tensor loss_reg{torch::zeros({1})};
        loss = loss.to(device);
        loss_reg = loss_reg.to(device);

        for (auto& batch : *data_loader)
        {
            optimizer->zero_grad();

            torch::Tensor data = torch::zeros({batch_size, output_dim*2 + 2}) ;
            torch::Tensor bb;

            // TODO fix the batch initialization
            for (int i; i< batch_size; i++)
            {
                bb = batch[i];
                // std::cout << "bb " << bb.size(1) << std::endl;
                data.slice(0, i, i+1)=batch[i];
            }

            // std::cout << "batch shape " << data.size(0) << " " << data.size(1) << std::endl;

            torch::Tensor snap_rec = autoencoder->forward(data.slice(1, 2, output_dim * 2 + 2).to(at::kFloat).to(device));

            // std::cout << "rec shape " << snap_rec.size(0) << " " << snap_rec.size(1) << std::endl;

            // std::cout << loss.item<float>() << std::endl;
            // std::cout << loss_reg.item<float>() << std::endl;
            auto loss1 = regularizerl1(device);
            auto loss2 = torch::nn::functional::mse_loss(snap_rec, data.slice(1, 2, output_dim * 2+ 2).to(device), torch::kSum);
            auto loss_tot = loss1 + loss2;

            loss_tot.backward();
            optimizer->step();

            loss_reg = loss_reg + loss1;
            loss = loss + loss2;
        }

        auto mean_loss = (loss + loss_reg) / (output_dim *2);
        std::cout << "Epoch, Loss: " << epoch << " , " << mean_loss.item<float>() << " Mean RMS: " << loss.item<float>() << " Reg: " << loss_reg.item<float>() << std::endl;

        torch::Tensor snap_rec = autoencoder->forward(train_snap_unscaled.to(at::kFloat).to(device));
        auto test_loss = torch::frobenius_norm(torch::abs(snap_rec-train_snap_unscaled))/test_loss_norm;
        std::cout << "Test, Loss: " << test_loss.item<float>() << std::endl;

    }

    std::cout << "Training finished!\n";
    // torch::save(autoencoder(), "./ITHACAoutput/Offline/NN/net.pt");

    autoencoder->eval();
    torch::NoGradGuard no_grad;

    torch::Tensor snap_rec = autoencoder->forward(train_snap_unscaled.to(at::kFloat).to(device));

    auto tensor_stacked = torch::cat({snap_rec.reshape({snap_rec.size(0), 60, 60, 2}), torch::zeros({snap_rec.size(0), 60, 60, 1})}, 3).reshape({snap_rec.size(0), 3600, 3});

    auto g = autoPtr<volVectorField>(new volVectorField(problem->L_Umodes[0]));

    for (auto i = 0; i < tensor_stacked.size(0); i++)
    {
        auto sliced = tensor_stacked.slice(0, i, i + 1).squeeze();
        auto push_forward = torch2Foam::torch2Field<vector>(sliced);
        g.ref().ref().field() = std::move(push_forward);
        save_field.append(g().clone());
    }

    std::cout << "SAVED" << std::endl;
    ITHACAstream::exportFields(save_field, "./Reconstructed","g");


}

void Embedding::loadData()
{
    auto ptrList = problem->Ufield;
    int Nrows = ptrList.size();
    int Ncols = ptrList[0].size() * 2;
    torch::Tensor train_snap_unscaled = torch::randn({Nrows, Ncols});

    std::cout << "Loading dataset ... " << Nrows << "x" << Ncols << std::endl;

    for (auto i = 0; i < ptrList.size(); i++)
    {
       auto imported = torch2Foam::field2Torch(ptrList[i]).squeeze();
        // std::cout << imported.size(0) << std::endl;
        train_snap_unscaled.slice(0, i, i + 1) = imported.reshape({output_dim, 3}).slice(1, 0, 2).reshape(output_dim*2);
    }
    std::cout << "Finished loading ... " << std::endl;

    // torch::Tensor train_snap_unscaled = ITHACAtorch::torch2Foam::ptrList2Torch(problem->Ufield);

    // initialize normalization parameters
    torch::Tensor min_sn = torch::min(train_snap_unscaled);
    torch::Tensor max_sn = torch::max(train_snap_unscaled);
    autoencoder->bias_inp  = min_sn;
    autoencoder->scale_inp = max_sn - min_sn;

    auto t_mu_mat = ITHACAstream::readMatrix("./ITHACAoutput/Offline/Training/mu_samples_mat.txt");
    auto t_mu_tensor = torch2Eigen::eigenMatrix2torchTensor(t_mu_mat);
    train_t_mu_snap_unscaled = torch::cat({t_mu_tensor.slice(0, 0, 2001), train_snap_unscaled}, 1);

    std::cout << "Dataset stacked shape: " << train_t_mu_snap_unscaled.size(0) << " x " << train_t_mu_snap_unscaled.size(1) << std::endl;

    data_set = autoPtr<SnapDataset>(new SnapDataset(train_t_mu_snap_unscaled));
}

torch::Tensor Embedding::lossResidual(torch::Tensor &rec_tensor, int idx_field, double mu, torch::Tensor & ref_tensor)
{
    // create foam field from reconstructed tensor
    auto g = autoPtr<volVectorField>(new volVectorField(_U0()));
    auto g0 = autoPtr<volVectorField>(new volVectorField(_U0()));

    auto tensor_stacked = torch::cat({snap_rec.reshape({snap_rec.size(0), 60, 60, 2}), torch::zeros({snap_rec.size(0), 60, 60, 1})}, 3).reshape({snap_rec.size(0), 3600, 3});
    auto push_forward = torch2Foam::torch2Field<vector>(tensor_stacked);

    auto tensor_stacked_ref = torch::cat({ref_tensor.reshape({ref_tensor.size(0), 60, 60, 2}), torch::zeros({ref_tensor.size(0), 60, 60, 1})}, 3).reshape({ref_tensor.size(0), 3600, 3});
    auto push_forward_ref = torch2Foam::torch2Field<vector>(tensor_stacked_ref);

    // add reference term
    g0.ref().ref().field() = std::move(push_forward_ref);
    g.ref().ref().field() = std::move(push_forward);
    g.ref() += mu * _U0() - g0();

    volVectorField& a_tmp = g();
    fvMesh& mesh = problem->_mesh();

    volVectorField& tmp = a_tmp.oldTime();
    tmp = problem->Ufield[idx_field];

    auto phi = linearInterpolate(a_tmp) & mesh.Sf();

    fvVectorMatrix resEqn(
        fvm::ddt(a_tmp) + 0.5 * fvm::div(phi, a_tmp) - fvm::laplacian(dimensionedScalar(dimViscosity, nu.value()), a_tmp));

    // convert the linear system to torch
    Eigen::VectorXd b;
    Eigen::MatrixXd A;
    foam2Eigen::fvMatrix2Eigen(resEqn, A, b);
    auto A_eigen = torch2Eigen:: eigenMatrix2torchTensor(A);
    auto b_eigen = torch2Eigen:: eigenMatrix2torchTensor(b);

    // TODO convert A_eigen to sparse matrix



}

// private method used only inside Embedding::forward. Return reference element of embedding s.t. initial embedding is mu * _U0()
autoPtr<volVectorField> Embedding::embedding_ref(const scalar mu)
{
    return autoPtr<volVectorField>(new volVectorField(mu * _U0() - _g0()));
}

autoPtr<volVectorField> Embedding::forward(const Eigen::VectorXd &x, const scalar mu)
{
    // Info << " #################### DEBUG
    // ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers_central.C,
    // line 109 #################### " << endl;

    // declare input of decoder of type IValue since the decoder is loaded from pytorch
    Eigen::MatrixXd input_matrix{x};

    input_matrix.resize(1, latent_dim);
    torch::Tensor input_tensor = torch2Eigen::eigenMatrix2torchTensor(std::move(input_matrix));
    input_tensor = input_tensor.reshape({1, latent_dim});
    input_tensor = input_tensor.set_requires_grad(true);

    torch::Tensor push_forward_tensor = autoencoder->decoder.forward(std::move(input_tensor));

    push_forward_tensor = push_forward_tensor * autoencoder->scale_inp + autoencoder->bias_inp;
    push_forward_tensor = push_forward_tensor .to(torch::kCPU);

    auto g = autoPtr<volVectorField>(new volVectorField(_U0()));

    // add the z component to the tensor as a zero {1, 60, 60} tensor and
    // reshape the tensor s.t. the components x,y,z of a single cell center are
    // contiguous in memory (this is necessary for torch2field method)
    auto tensor_stacked = torch::cat({push_forward_tensor.reshape({2, 60, 60}), torch::zeros({1, 60, 60})}, 0).reshape({3, -1}).transpose(0, 1).contiguous();

    auto push_forward = torch2Foam::torch2Field<vector>(tensor_stacked);

    // add reference term
    g.ref().ref().field() = std::move(push_forward);
    g.ref() += embedding_ref(mu).ref();

    // save_field.append(g().clone());
    // if (counter == 10) {
    //     std::cout << "SAVED" << std::endl;
    //     ITHACAstream::exportFields(save_field, "./Forwarded","g");
    // }

    return g;
}

/* Since torch::autograd::jacobian is not implemented in libtorch yet, this is
one of among the possible ways to compute the full jacobian with
torch::autograd::grad.The drawback is that 7200-by-4 repeated inputs are
forwarded to obtain an output of dimension 7200-by-7200 and then a costly
backward is computed. Since this operation could require a lot of GPU memory,
the evaluation of the components of the jacobian is split in 2 batches of
3600.*/
// autoPtr<Eigen::MatrixXd> Embedding::jacobian(const Eigen::VectorXd &x, const scalar mu)
// {
//     // dimension of degrees of freedom associated to x and y components
//     int jacobian_out_dim = output_dim * 2; // 7200
//     Eigen::MatrixXd input_matrix{x};

//     input_matrix.resize(1, latent_dim);
//     torch::Tensor input_tensor = torch2Eigen::eigenMatrix2torchTensor(std::move(input_matrix));
//     input_tensor = input_tensor.reshape({1, latent_dim}).set_requires_grad(true);

//     // compute the jacobian with batches of 3600 for a total of 7200 components.
//     // Since torch::autograd::
//     auto input_repeated = input_tensor.repeat({3600, 1});
//     input_repeated = input_repeated.set_requires_grad(true);

//     // declare input of decoder of type IValue since the decoder is loaded from
//     // pytorch. The tensor inputs of the decoder must be of type at::kFloat (not double)
//     std::vector<torch::jit::IValue> input_jac;
//     input_jac.push_back(input_repeated.to(at::kFloat).to(torch::kCUDA));

//     // term to multiply with matrix-to-matrix product with the Jacobian of the
//     // net: since it is the identity 7200-by-7200 matrix we obtainexactly the Jacobian.
//     auto grad_output = torch::eye(output_dim * 2);

//     // initialize the jacobian of the decoder of size jacobian_out_dim-by-latent_dim
//     torch::Tensor forward_tensor = decoder->forward(input_jac).toTensor().squeeze();
//     auto J = torch::ones({jacobian_out_dim, latent_dim});

//     // compute the jacobian with batches of 3600 for a total of 7200 components
//     for(int i=0; i<2; i++)
//     {
//         auto grad_component = grad_output.slice(0, i*3600, (1+i)*3600).to(torch::kCUDA);
//         forward_tensor.backward(grad_component, true);

//         auto gradient = torch::autograd::grad({forward_tensor},
//                                           {input_repeated},
//                                           /*grad_outputs=*/{grad_component},
//                                           /*retain_graph=*/true,
//                                           /*create_graph=*/true);

//         J.slice(/*dim*/0, i*3600, (1+i)*3600) = gradient[0].detach();
//     }

//     auto grad_eigen = torch2Eigen::torchTensor2eigenMatrix<double>(J);
//     auto dg = autoPtr<Eigen::MatrixXd>(new Eigen::MatrixXd(std::move(grad_eigen)));

//     return dg;
// }

// std::pair<autoPtr<volVectorField>, autoPtr<Eigen::MatrixXd>> Embedding::forward_with_gradient(const Eigen::VectorXd &x, const scalar mu)
// {
//     auto g = forward(x, mu);
//     auto dg = jacobian(x, mu);
//     return std::make_pair(g, dg);
// }

// Operator to evaluate the residual
int newton_nmlspg_burgers::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
{
    std::cout << " residual, x = " << x.transpose() << endl;

    auto g = embedding->forward(x.head(Nphi_u), mu);
    volVectorField& a_tmp = g();
    fvMesh& mesh = problem->_mesh();


    auto a_old = g_old();
    volVectorField& tmp = a_tmp.oldTime();
    tmp = a_old;

    auto phi = linearInterpolate(a_tmp) & mesh.Sf();

    fvVectorMatrix resEqn(
        fvm::ddt(a_tmp) + 0.5 * fvm::div(phi, a_tmp) - fvm::laplacian(dimensionedScalar(dimViscosity, nu.value()), a_tmp));

    // resEqn.solve();
    a_tmp.field() = resEqn.residual();

    fvec = Foam2Eigen::field2Eigen(a_tmp).col(0).head(this->embedding->output_dim * 2);

    // this->embedding->save_field.append(a_tmp.clone());

    // if (this->embedding->counter == 10) {
    //     std::cout << "SAVED" << std::endl;
    //     ITHACAstream::exportFields(this->embedding->save_field, "./RESIDUAL", "res");
    // }

    // this->embedding->counter++;

    Info << " residual norm: " << fvec.norm() << endl;

    return 0;
}

// * * * * * * * * * * * * * * * Solve Functions  * * * * * * * * * * * * * //
void NonlinearReducedBurgers::solveOnline(Eigen::MatrixXd mu, int startSnap)
{

    M_Assert(exportEvery >= dt,
             "The time step dt must be smaller than exportEvery.");
    M_Assert(storeEvery >= dt,
             "The time step dt must be smaller than storeEvery.");
    // M_Assert(exportResidual >= dt,
    //          "The time step dt must be smaller than exportEvery.");
    M_Assert(ITHACAutilities::isInteger(storeEvery / dt) == true,
             "The variable storeEvery must be an integer multiple of the time step dt.");
    M_Assert(ITHACAutilities::isInteger(exportEvery / dt) == true,
             "The variable exportEvery must be an integer multiple of the time step dt.");
    M_Assert(ITHACAutilities::isInteger(exportResidual / dt) == true,
             "The variable storeEvery must be an integer multiple of the time step dt.");
    M_Assert(ITHACAutilities::isInteger(exportEvery / storeEvery) == true,
             "The variable exportEvery must be an integer multiple of the variable storeEvery.");
    // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers_central.C, line 264 #################### " << endl;
    int numberOfStores = round(storeEvery / dt);

    // numberOfResiduals defaults to 0 and in that case no residual is saved
    int numberOfResiduals = round(exportResidual / dt);

    // Counter of the number of online solutions saved, accounting also time as parameter
    int counter2 = 0;

    // Set number of online solutions
    int Ntsteps = static_cast<int>((finalTime - tstart) / dt);
    int onlineSizeTimeSeries = static_cast<int>(Ntsteps / numberOfStores);

    // resize the online solution list with the length of n_parameters times
    // length of the time series
    online_solution.resize((mu.cols()) * (onlineSizeTimeSeries));
    // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers_central.C, line 280 #################### " << endl;
    // Iterate online solution for each parameter saved row-wise in mu
    for (int n_param = 0; n_param < mu.cols(); n_param++)
    {

        // Set the initial time
        time = tstart;

        // Counter of the number of saved time steps for the present parameter with index n_param
        int counter = 0;
        int nextStore = 0;

        // residual export for hyper-reduction counter
        int counterResidual = 1;
        // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers_central.C, line 294 #################### " << endl;
        // Create and resize the solution vector (column vector)
        y.resize(Nphi_u, 1);
        y = embedding->latent_initial.transpose();
        newton_object.g_old = embedding->forward(y, mu(0, n_param));
        // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers_central.C, line 300 #################### " << endl;
        auto tmp = newton_object.g_old();
        // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers_central.C, line 302 #################### " << endl;
        uRecFields.append(tmp.clone());
        // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers_central.C, line 303 #################### " << endl;
        // Set some properties of the newton object
        newton_object.mu = mu(0, n_param);
        newton_object.nu = nu;
        newton_object.dt = dt;
        newton_object.tauU = tauU;

        // Create vector to store temporal solution and save initial condition
        // as first solution
        // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers_central.C, line 313 #################### " << endl;
        Eigen::MatrixXd tmp_sol(Nphi_u + 1, 1);
        tmp_sol(0) = time;
        // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers_central.C, line 316 #################### " << endl;
        tmp_sol.col(0).tail(y.rows()) = y;
        // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers_central.C, line 318 #################### " << endl;
        online_solution[counter2] = tmp_sol;
        counter2++;
        counter++;
        nextStore += numberOfStores;

        // Create nonlinear solver object
        // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers_central.C, line 321 #################### " << endl;
        // Create nonlinear solver object
        Eigen::NumericalDiff<newton_nmlspg_burgers, Eigen::Central> numDiffobject(newton_object, 1.e-05);
        Eigen::LevenbergMarquardt<decltype(numDiffobject)> lm(numDiffobject);

        lm.parameters.factor = 100; //step bound for the diagonal shift, is this related to damping parameter, lambda?
        lm.parameters.maxfev = 5000;//max number of function evaluations
        lm.parameters.xtol = 1.49012e-20; //tolerance for the norm of the solution vector
        lm.parameters.ftol = 1.49012e-20; //tolerance for the norm of the vector function
        lm.parameters.gtol = 0; // tolerance for the norm of the gradient of the error vector
        lm.parameters.epsfcn = 0; //error precision

        // Set output colors for fancy output
        Color::Modifier red(Color::FG_RED);
        Color::Modifier green(Color::FG_GREEN);
        Color::Modifier def(Color::FG_DEFAULT);

        time = time + dt;

        while (time < finalTime)
        {
            Eigen::LevenbergMarquardtSpace::Status ret = lm.minimize(y);

            std::cout << "LM finished with status: " << ret << std::endl;

            Info << " minimum: x=(" << y(0) << " " << y(1) << " " << y(2) << " " << y(3) << ")" << endl;

            Eigen::VectorXd res(2 * numDiffobject.embedding->output_dim);
            res.setZero();

            // update the old solution for the evaluation of the residual
            numDiffobject(y, res);
            numDiffobject.g_old = embedding->forward(y, mu(0, n_param));
            auto tmp = numDiffobject.g_old();
            uRecFields.append(tmp.clone());

            std::cout << "################## Online solve N° " << counter << " ##################" << std::endl;
            Info << "Time = " << time << endl;

            if (res.norm() < 1e-5)
            {
                std::cout << green << "|F(x)| = " << res.norm() << " - Minimun reached in " << lm.iter << " iterations " << def << std::endl
                          << std::endl;
            }
            else
            {
                std::cout << red << "|F(x)| = " << res.norm() << " - Minimun reached in " << lm.iter << " iterations " << def << std::endl
                          << std::endl;
            }

            tmp_sol(0) = time;
            tmp_sol.col(0).tail(y.rows()) = y;

            if (counter == nextStore)
            {
                if (counter2 >= online_solution.size())
                {
                    online_solution.append(tmp_sol);
                }
                else
                {
                    online_solution[counter2] = tmp_sol;
                }

                nextStore += numberOfStores;
                counter2++;
            }

            if (numberOfResiduals > 0)
            {
                if (counterResidual == numberOfResiduals)
                {
                    volScalarField res_to_be_saved = numDiffobject.g_old().component(0);

                    // Eigen::VectorXd zeros = Eigen::VectorXd::Zero(numDiffobject.embedding->output_dim);
                    // Eigen::VectorXd res_extended(res.size() + zeros.size());
                    // res_extended << res, zeros;
                    Eigen::VectorXd res_component = res.head(this->embedding->output_dim);
                    res_to_be_saved = Foam2Eigen::Eigen2field(res_to_be_saved, res_component);
                    residualsList.append(res_to_be_saved);
                    counterResidual = 1;
                }

                counterResidual++;
            }

            counter++;
            time = time + dt;
        }
    }
}

// * * * * * * * * * * * * * * *  Evaluation  * * * * * * * * * * * * * //

void NonlinearReducedBurgers::reconstruct(bool exportFields, fileName folder)
{
    if (exportFields)
    {
        mkDir(folder);
        ITHACAutilities::createSymLink(folder);
    }

    int counter = 0;
    int nextwrite = 0;
    List<Eigen::MatrixXd> CoeffU;
    List<double> tValues;
    CoeffU.resize(0);
    tValues.resize(0);
    int exportEveryIndex = round(exportEvery / storeEvery);
    std::vector<torch::jit::IValue> input;

    for (int i = 0; i < online_solution.size(); i++)
    {
        if (counter == nextwrite)
        {
            Eigen::MatrixXd currentUCoeff;
            currentUCoeff = online_solution[i].block(1, 0, Nphi_u, 1);
            CoeffU.append(currentUCoeff);
            nextwrite += exportEveryIndex;
            double timeNow = online_solution[i](0, 0);
            tValues.append(timeNow);
        }

        counter++;
    }

    if (exportFields)
    {
        ITHACAstream::exportFields(uRecFields, folder,
                                   "uRec");
    }
}

//TODO: not implemented yet
void NonlinearReducedBurgers::reconstruct(bool exportFields, fileName folder, Eigen::MatrixXd redCoeff)
{
    if (exportFields)
    {
        mkDir(folder);
        ITHACAutilities::createSymLink(folder);
    }

    int counter = 0;
    int nextwrite = 0;
    List<Eigen::MatrixXd> CoeffU;
    List<double> tValues;
    CoeffU.resize(0);
    tValues.resize(0);
    int exportEveryIndex = round(exportEvery / storeEvery);

    for (int i = 0; i < redCoeff.rows(); i++)
    {
        if (counter == nextwrite)
        {
            Eigen::MatrixXd currentUCoeff(Nphi_u, 1);

            currentUCoeff.col(0) = redCoeff.row(i).tail(Nphi_u).transpose();

            CoeffU.append(currentUCoeff);
            nextwrite += exportEveryIndex;
            double timeNow = redCoeff(i, 0);
            tValues.append(timeNow);
        }

        counter++;
    }

    volVectorField uRec("uRec", Umodes[0] * 0);

    uRecFields = problem->L_Umodes.reconstruct(uRec, CoeffU, "uRec");

    if (exportFields)
    {
        ITHACAstream::exportFields(uRecFields, folder,
                                   "uRec");
    }
}

Eigen::MatrixXd NonlinearReducedBurgers::setOnlineVelocity(Eigen::MatrixXd vel)
{
    assert(problem->inletIndex.rows() == vel.rows() && "Imposed boundary conditions dimensions do not match given values matrix dimensions");
    Eigen::MatrixXd vel_scal;
    vel_scal.resize(vel.rows(), vel.cols());

    for (int k = 0; k < problem->inletIndex.rows(); k++)
    {
        int p = problem->inletIndex(k, 0);
        int l = problem->inletIndex(k, 1);
        scalar area = gSum(problem->liftfield[0].mesh().magSf().boundaryField()[p]);
        scalar u_lf = gSum(problem->liftfield[k].mesh().magSf().boundaryField()[p] *
                           problem->liftfield[k].boundaryField()[p])
                          .component(l) /
                      area;

        for (int i = 0; i < vel.cols(); i++)
        {
            vel_scal(k, i) = vel(k, i) / u_lf;
        }
    }

    return vel_scal;
}

void NonlinearReducedBurgers::trueProjection(fileName folder)
{
    List<Eigen::MatrixXd> CoeffU;
    CoeffU.resize(0);

    for (int n_index = 0; n_index < problem->Ufield.size(); n_index++)
    {
        Eigen::MatrixXd currentUCoeff(Nphi_u, 1);

        currentUCoeff.col(0) = ITHACAutilities::getCoeffs(problem->Ufield[n_index], Umodes);

        CoeffU.append(currentUCoeff);
    }

    volVectorField uRec("uRec", Umodes[0] * 0);

    uRecFields = problem->L_Umodes.reconstruct(uRec, CoeffU, "uRec");

    ITHACAstream::exportFields(uRecFields, folder, "uTrueProjection");
}