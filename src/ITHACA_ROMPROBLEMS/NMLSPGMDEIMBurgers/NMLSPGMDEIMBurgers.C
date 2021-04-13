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
/// Source file of the NMLSPGMDEIMBurgers class

#include <torch/script.h>
#include <torch/torch.h>
#include "torch2Eigen.H"
#include "torch2Foam.H"
#include "Foam2Eigen.H"
#include "ITHACAstream.H"
#include <chrono>
#include "cnpy.H"
#include <cmath>

#include "NMLSPGMDEIMBurgers.H"

using namespace ITHACAtorch;

// * * * * * * * * * * * * * * * Embeddings MDEIM * * * * * * * * * * * * * * * * //
EmbeddingMDEIM::EmbeddingMDEIM(int dim, fileName decoder_path, volVectorField &U0, Eigen::MatrixXd lat_init) : latent_dim{dim}, latent_initial{lat_init}
{
    decoder = autoPtr<torch::jit::script::Module>(new torch::jit::script::Module(torch::jit::load(decoder_path)));
    _U0 = autoPtr<volVectorField>(new volVectorField(U0));

    // declare input of decoder of type IValue since the decoder is loaded from pytorch
    std::vector<torch::jit::IValue> input;
    torch::Tensor latent_initial_tensor = torch2Eigen::eigenMatrix2torchTensor(latent_initial);

    // the tensor inputs of the decoder must be of type at::kFloat (not double)
    input.push_back(latent_initial_tensor.to(at::kFloat).to(torch::kCUDA));

    std::cout << "LATENT INITIAL" << latent_initial_tensor << std::endl;
    torch::Tensor forwarded = decoder->forward(std::move(input)).toTensor().to(torch::kCPU);
    initial_mdeim = torch2Eigen::torchTensor2eigenMatrix<double>(forwarded).transpose();
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 64 # " << endl;
}

Eigen::VectorXd EmbeddingMDEIM::forward(const Eigen::VectorXd &x, const scalar mu)
{
    // declare input of decoder of type IValue since the decoder is loaded from pytorch
    std::vector<torch::jit::IValue> input;
    Eigen::MatrixXd input_matrix{x};

    input_matrix.resize(1, latent_dim);
    torch::Tensor input_tensor = torch2Eigen::eigenMatrix2torchTensor(std::move(input_matrix));
    input_tensor = input_tensor.reshape({1, latent_dim});
    input_tensor = input_tensor.set_requires_grad(true);

    // the tensor inputs of the decoder must be of type at::kFloat (not double)
    input.push_back(input_tensor.to(at::kFloat).to(torch::kCUDA));
    torch::Tensor push_forward_tensor = decoder->forward(std::move(input)).toTensor().to(torch::kCPU);

    Eigen::MatrixXd push_forward = torch2Eigen::torchTensor2eigenMatrix<double>(push_forward_tensor).transpose();
    // std::cout << "push_forward: " << push_forward.rows() << " x " <<  push_forward.cols() << std::endl;
    // add reference term
    return push_forward;
}


// * * * * * * * * * * * * * * * newton_burgers * * * * * * * * * * * * * * * * //
// private method used only inside EmbeddingMDEIM::forward. Return reference element of embedding s.t. initial embedding is mu * _U0()
Eigen::VectorXd newton_nmlspg_mdeim_burgers::embedding_ref(const scalar mu) const
{
    // std::cout << "U0_mdeim: " << U0_mdeim.rows() << " x " <<  U0_mdeim.cols() << std::endl;
//     std::cout << "initial_mdeim: " << embedding->initial_mdeim.rows() << " x " <<  embedding->initial_mdeim.cols() << std::endl;
    return mu * U0_mdeim - embedding->initial_mdeim;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
newton_nmlspg_mdeim_burgers::onlineCoeffsAB(double mu,PtrList<volVectorField> listA,
                                                      PtrList<volVectorField> listB) const
{
    int mpB = total_mp-listA.size();
    Eigen::VectorXd AA(listA.size());
    Eigen::VectorXd BA(listA.size());
    Eigen::VectorXd AB(mpB);
    Eigen::VectorXd BB(mpB);

    for (int i = 0; i < listA.size(); i++)
    {
        Eigen::SparseMatrix<double> Mr;
        Eigen::VectorXd br;
        fvVectorMatrix Aof = evaluate_expression(listA[i], listAold[i], mu);
        Foam2Eigen::fvMatrix2Eigen(Aof, Mr, br);
        int ind_row = mdeim->localMagicPointsA[i].first() + mdeim->xyz_A[i].first() *
                        listA[i].size();
        int ind_col = mdeim->localMagicPointsA[i].second() + mdeim->xyz_A[i].second() *
                        listA[i].size();
        AA(i) = Mr.coeffRef(ind_row, ind_col);
        BA(i) = br(ind_col);
    }

    for (int i = 0; i < listB.size(); i++)
    {
        Eigen::SparseMatrix<double> Mr;
        Eigen::VectorXd br;
        fvVectorMatrix Aof = evaluate_expression(listB[i], listBold[i], mu);
        Foam2Eigen::fvMatrix2Eigen(Aof, Mr, br);
        int ind_row = mdeim->localMagicPointsB[i] + mdeim->xyz_B[i] * listB[i].size();
        AB(i) = Mr.coeffRef(ind_row, ind_row);
        BB(i) = br(ind_row);
    }

    return std::make_tuple(AA, BA, AB, BB);
}

Eigen::MatrixXd newton_nmlspg_mdeim_burgers::onlineCoeffsA(double mu, PtrList<volVectorField> listA) const
{
    Eigen::MatrixXd theta(listA.size(), 1);

    for (int i = 0; i < listA.size(); i++)
    {
        Eigen::SparseMatrix<double> Mr;
        Eigen::VectorXd br;
        fvVectorMatrix Aof = evaluate_expression(listA[i], listAold[i], mu);
        Foam2Eigen::fvMatrix2Eigen(Aof, Mr, br);
        int ind_row = mdeim->localMagicPointsA[i].first() + mdeim->xyz_A[i].first() *
                        listA[i].size();
        int ind_col = mdeim->localMagicPointsA[i].second() + mdeim->xyz_A[i].second() *
                        listA[i].size();
        theta(i) = Mr.coeffRef(ind_row, ind_col);
    }

    return theta;
}

Eigen::MatrixXd newton_nmlspg_mdeim_burgers::onlineCoeffsB(double mu, PtrList<volVectorField> listB) const
{
    Eigen::MatrixXd theta(listB.size(), 1);

    for (int i = 0; i < listB.size(); i++)
    {
        Eigen::SparseMatrix<double> Mr;
        Eigen::VectorXd br;
        fvVectorMatrix Aof = evaluate_expression(listB[i], listBold[i], mu);
        Foam2Eigen::fvMatrix2Eigen(Aof, Mr, br);
        int ind_row = mdeim->localMagicPointsB[i] + mdeim->xyz_B[i] * listB[i].size();
        theta(i) = br(ind_row);
    }

    return theta;
}

fvVectorMatrix newton_nmlspg_mdeim_burgers::evaluate_expression(volVectorField& U, const volVectorField& U_old, double mu) const
{
    fvMesh& mesh  =  const_cast<fvMesh&>(U.mesh());
    auto phi = linearInterpolate(U) & mesh.Sf();

    volVectorField &tmp = U.oldTime();
    tmp = U_old;

    fvVectorMatrix resEqn(
        fvm::ddt(U) + 0.5 * fvm::div(phi, U) -
        fvm::laplacian(dimensionedScalar(dimViscosity, nu.value()), U));

    return resEqn;
}

void newton_nmlspg_mdeim_burgers::init_old(Eigen::VectorXd& U_old)
{
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 190 # " << U_old.rows() << endl;
    int index = 0;
    for (int i = 0; i < mdeim->fieldsA.size(); i++)
    {
        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 160 # A " << index << " " << i << " " << mdeim->fieldsA[i].size()<< endl;
        int n = mdeim->fieldsA[i].size();
        volVectorField field{mdeim->fieldsA[i]};
        Eigen::VectorXd localVectorx = U_old.segment(index, n);
        Eigen::VectorXd localVectory = U_old.segment(index+total_submeshes_points, n);

        for (int j = 0; j < mdeim->fieldsA[i].size(); j++)
        {
            field[j][0] = localVectorx(j);
            field[j][1] = localVectory(j);
        }

        listAold.append(field);
        index += n;
    }

    // continuing from index value of previous loop
    int k{0};

    for (int i = 0; i < mdeim->fieldsB.size(); i++)
    {
        if (matrixBindices(i)==1)
        {
            // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 160 # B " << index << " "<< i << " " << mdeim->fieldsB[i].size()<< endl;
            int n = mdeim->fieldsB[i].size();
            volVectorField field{mdeim->fieldsB[i]};
            Eigen::VectorXd localVectorx = U_old.segment(index, n);
            Eigen::VectorXd localVectory = U_old.segment(index+total_submeshes_points, n);

            for (int j = 0; j < mdeim->fieldsB[i].size(); j++)
            {
                field[j][0] = localVectorx(j);
                field[j][1] = localVectory(j);
            }
            listBold.append(field);
            index += n;
            k++;
        }
    }
}

void newton_nmlspg_mdeim_burgers::rec_field(Eigen::VectorXd& vec, volVectorField& fieldRec, word name)
{
    fieldRec = 0*fieldRec;
    int k{0};
    for (int i; i<mdeim->magicPointsA.size(); i++)
    {
        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 241 # " << mdeim->xyz_A[i].second() << " " << mdeim->magicPointsA[i].second() << endl;
        fieldRec[mdeim->magicPointsA[i].second()][mdeim->xyz_A[i].second()] = 1;//vec[k];
        k++;
    }
    for (int i; i<mdeim->magicPointsB.size(); i++)
    {
        if (matrixBindices(i)==1)
        {
            // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 249 # " << mdeim->xyz_B[i] << " " << mdeim->magicPointsB[i] << endl;
            fieldRec[mdeim->magicPointsB[i]][mdeim->xyz_B[i]] = 1;//vec[k];
            k++;
        }
    }
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 251 # " << k << endl;

    PtrList<volVectorField> saved;
    saved.append(fieldRec);
    ITHACAstream::exportFields(saved, "NMReconstructed", name+std::to_string(counterRec));
    counterRec++;

}

// TODO fix clumsy loops and indices
Eigen::VectorXd newton_nmlspg_mdeim_burgers::restrict_decoder()
{
    Eigen::VectorXd g_restricted;
    g_restricted.resize(total_mp);

    for (int i = 0; i < mdeim->fieldsA.size(); i++)
    {
        Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 271 # " << i << endl;
        Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 272 # " << mdeim->xyz_A[i].second() << " " << mdeim->localMagicPointsA[i].second() << endl;
        Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 272 # " << listAold[i][mdeim->xyz_A[i].second()][mdeim->localMagicPointsA[i].second()] << endl;
        g_restricted(i) = listAold[i][mdeim->xyz_A[i].second()][mdeim->localMagicPointsA[i].second()];

    }

    int k{mdeim->fieldsA.size()};
    int j{0};
    for (int i = 0; i < mdeim->fieldsB.size(); i++)
    {
        if (matrixBindices(i)==1)
        {
            Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 281 # " << i << " " << k << " " << j << endl;
            Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 285 # " << mdeim->xyz_B[i] << " " << mdeim->localMagicPointsB[i] << endl;
            Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 283 # " << listBold[j][mdeim->xyz_B[i]][mdeim->localMagicPointsB[i]] << endl;
            g_restricted(k) = listBold[j][mdeim->xyz_B[i]][mdeim->localMagicPointsB[i]];
            k++;
            j++;
        }
    }

    return g_restricted;
}

// Operator to evaluate the residual
int newton_nmlspg_mdeim_burgers::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
{
    // evaluate the compressed decoder at the new latent values x
    // the dimension of g is total_submeshes_points * 2 ( *2 stands for x, y components)
    auto start = std::chrono::system_clock::now();
    Eigen::VectorXd g = embedding->forward(x, mu) + embedding_ref(mu);
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << " residual, x = " << x.transpose() << " Elapsed in: " << elapsed.count() << " microseconds." << std::endl;

    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 152 # " << g.rows() << " " << g.cols() << endl;


    start = std::chrono::system_clock::now();
    // intialize listA of submeshes fields needed to evaluate the online coeffs
    // of A and b (the stencil depends on numerical scheme chosen)
    PtrList<volVectorField> listA;
    PtrList<volVectorField> listB;

    // evaluate g at the magicPoints in g_restrictedA and g_restrictedB:
    // discarding the other local magic points of the submesh
    Eigen::VectorXd g_restrictedA;
    g_restrictedA.resize(mdeim->fieldsA.size());
    Eigen::VectorXd g_restrictedB;
    g_restrictedB.resize(total_mp-mdeim->fieldsA.size());

    // TODO create auxiliary function for these 2 for loops
    int index = 0;
    for (int i = 0; i < mdeim->fieldsA.size(); i++)
    {
        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 160 # A " << index << " " << i << " " << mdeim->fieldsA[i].size()<< endl;
        int n = mdeim->fieldsA[i].size();
        volVectorField field{mdeim->fieldsA[i]};
        Eigen::VectorXd localVectorx = g.segment(index, n);
        Eigen::VectorXd localVectory = g.segment(index+total_submeshes_points, n);

        for (int j = 0; j < mdeim->fieldsA[i].size(); j++)
        {
            field[j][0] = localVectorx(j);
            field[j][1] = localVectory(j);
        }

        listA.append(field);

        // initialize g_restricted to magicPointsA only
        g_restrictedA(i) = field[mdeim->localMagicPointsA[i].second()][mdeim->xyz_A[i].second()];

        index += n;
    }

    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 179 # " << endl;
    // continuing from index value of previous loop
    int k{0};

    for (int i = 0; i < mdeim->fieldsB.size(); i++)
    {
        if (matrixBindices(i)==1)
        {
            // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 160 # B " << index << " "<< i << " " << mdeim->fieldsB[i].size()<< endl;
            int n = mdeim->fieldsB[i].size();
            volVectorField field{mdeim->fieldsB[i]};
            // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 237 # " << endl;
            Eigen::VectorXd localVectorx = g.segment(index, n);
            // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 239 # " << endl;
            Eigen::VectorXd localVectory = g.segment(index+total_submeshes_points, n);
            // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 241 # " << endl;

            for (int j = 0; j < mdeim->fieldsB[i].size(); j++)
            {
                field[j][0] = localVectorx(j);
                // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 246 # " << endl;
                field[j][1] = localVectory(j);
                // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 248 # " << j <<endl;
            }
            listB.append(field);
            index += n;
            // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 252 # " << mdeim->xyz_B[i] <<endl;
            // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 253 # " << mdeim->localMagicPointsB[i] <<endl;
            // initialize g_restricted to magicPointsA only
            g_restrictedB(k) = field[mdeim->localMagicPointsB[i]][mdeim->xyz_B[i]];
            // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 255 # " << endl;
            k++;
        }
    }
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << " Combine system elapsed in: " << elapsed.count() << " microseconds." << std::endl;


    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 205 # " << g_restrictedA.size() << " " << g_restrictedB.size() <<endl;

    // AA, BB are interpolated, BA, AB are obtained through least-squares
    auto [AA, BA, AB, BB] = onlineCoeffsAB(mu, listA, listB);
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 266 # " << AA.rows() << " " << AA.cols() << endl;
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 266 # " << BA.rows() << " " << BA.cols() << endl;
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 266 # " << AB.rows() << " " << AB.cols() << endl;
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 266 # " << BB.rows() << " " << BB.cols() << endl;
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 266 # " << g_restrictedA.rows() << " " << g_restrictedA.cols() << endl;
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 266 # " << g_restrictedB.rows() << " " << g_restrictedB.cols() << endl;
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 266 # " << pinvAB.rows() << " " << pinvAB.cols() << endl;
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 266 # " << pinvBA.rows() << " " << pinvBA.cols() << endl;
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 274 # " << mdeim->fieldsA.size() << " " << total_mp-mdeim->fieldsA.size()<< endl;

    // solve the block system, for the A and B magicPoints separately
    fvec.head(mdeim->fieldsA.size()) =  AA.asDiagonal() * g_restrictedA - pinvBA * BA;
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 268 # " << endl;
    fvec.tail(total_mp-mdeim->fieldsA.size()) =  (pinvAB * AB).asDiagonal() * g_restrictedB - BB;

    return 0;
}

// Using the Eigen library, using the SVD decomposition method to solve the matrix pseudo-inverse, the default error er is 0
Eigen::MatrixXd pinv_eigen_based(Eigen::MatrixXd & origin, const float er = 0) {
    // perform svd decomposition
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 276 # " << endl;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_holder(origin,
                                                 Eigen::ComputeThinU |
                                                 Eigen::ComputeThinV);
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 280 # " << endl;
    // Build SVD decomposition results
    Eigen::MatrixXd U = svd_holder.matrixU();
    Eigen::MatrixXd V = svd_holder.matrixV();
    Eigen::MatrixXd D = svd_holder.singularValues();

    // Build the S matrix
    Eigen::MatrixXd S(V.cols(), U.cols());
    S.setZero();

    for (unsigned int i = 0; i < D.size(); ++i) {

        if (D(i, 0) > er) {
            S(i, i) = 1 / D(i, 0);
        } else {
            S(i, i) = 0;
        }
    }

    // pinv_matrix = V * S * U^T
    return V * S * U.transpose();
}

//! only offline
void newton_nmlspg_mdeim_burgers::localAB2generalizedCoord(Eigen::VectorXd matrixBindices)
{
    Eigen::MatrixXd matB;
    matB.resize(mdeim->magicPointsA.size(), (matrixBindices.array() > 0).count());

    int k{0};
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 311 # " << mdeim->magicPointsA.size() << " "<<  mdeim->magicPointsB.size() << endl;
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 280 # " << mdeim->UB.cols() << endl;
    // std::cout << " # DEBUG NMLSPGMDEIMBurgers.C, line 281 # " << matrixBindices.rows() << std::endl;
    for (int j = 0; j < mdeim->magicPointsB.size(); j++)
    {
        if (matrixBindices(j)==1)
        {
            // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 284 # " << k << endl;
            for (int i = 0; i < mdeim->magicPointsA.size(); i++)
            {
                int ind_col = mdeim->magicPointsA[i].second();
                matB.coeffRef(i, k) = mdeim->UB.coeffRef(ind_col, j);
            }
            k++;
        }
    }
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 293 # " << endl;
    Eigen::MatrixXd matA;
    matA.resize((matrixBindices.array() > 0).count(), mdeim->magicPointsA.size());
    k = 0;

    for (int j = 0; j < mdeim->magicPointsA.size(); j++)
    {
        for (int i = 0; i < mdeim->magicPointsB.size(); i++)
        {
            if (matrixBindices(i)==1)
            {
                int ind_col = mdeim->magicPointsB[i];
                // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 304 # " << k <<  " " << mdeim->UA[j].coeffRef(ind_col, ind_col) << endl;

                // TODO check
                matA.coeffRef(k, j) = mdeim->UA[j].coeffRef(ind_col, ind_col);
                k++;
            }
        }
        k=0;
    }

    std::cout << " A: " << matA << std::endl;
    std::cout << " B: " << matB << std::endl;

    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 310 # " << matA.rows() << " " << matA.cols() << endl;
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 310 # " << matB.rows() << " " << matB.cols() << endl;
    auto pA = pinv_eigen_based(matA);
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 316 # " << pA.rows() << " " << pA.cols() << endl;
    auto pB = pinv_eigen_based(matB);
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 347 # " << pB.rows() << " " << pB.cols()<< endl;
    pinvAB = matA * pA;
    pinvBA = matB * pB;
}

 // TODO check consistency of indexes
Eigen::VectorXd newton_nmlspg_mdeim_burgers::projectVector(Eigen::VectorXd full_vector) const
{
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 238 # " << total_submeshes_points << endl;
    Eigen::VectorXd hr_vector(2*total_submeshes_points, 1);
    int scalar_dim = full_vector.size()/3;
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 241 # " << scalar_dim << endl;
    int index = 0;

    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 243 # " << mdeim->fieldsA.size() << " " << mdeim->fieldsB.size() << endl;
    for (int i = 0; i < mdeim->fieldsA.size(); i++)
    {
        int n = mdeim->fieldsA[i].size();
        for (int j = 0; j < mdeim->submeshListA[i].cellMap().size(); j++)
        {
            // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 249 # " << i << " " << mdeim->submeshListA[i].cellMap().size() << " " << j << endl;
            hr_vector(index + j) = full_vector(mdeim->submeshListA[i].cellMap()[j]);
            hr_vector(index + total_submeshes_points + j) = full_vector(mdeim->submeshListA[i].cellMap()[j]+scalar_dim);
        }
        index += n;
        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 248 # " << index << endl;
    }

    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 254 # " << mdeim->fieldsB.size() << endl;
    // continuing from index value of previous loop
    for (int i = 0; i < mdeim->fieldsB.size(); i++)
    {
        if (matrixBindices(i)==1)
        {
            int n = mdeim->fieldsB[i].size(); // multplication by 2 because x, y coord
            for (int j = 0; j < mdeim->submeshListB[i].cellMap().size(); j++)
            {
                // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 257 # " << i << " " << mdeim->submeshListB[i].cellMap().size() << " " << j << endl;
                hr_vector(index + j) = full_vector(mdeim->submeshListB[i].cellMap()[j]);
                hr_vector(index + total_submeshes_points + j) = full_vector(mdeim->submeshListB[i].cellMap()[j]+scalar_dim);
            }
            index += n;
            // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 262 # " << index << endl;
        }

    }

    return hr_vector;
}

// * * * * * * * * * * * * * * * NMLSPGMDEIMBurgers * * * * * * * * * * * * * * * * //
// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

NMLSPGMDEIMBurgers::NMLSPGMDEIMBurgers()
{
    para = ITHACAparameters::getInstance();
}

NMLSPGMDEIMBurgers::NMLSPGMDEIMBurgers(Burgers &FOMproblem, fileName decoder_path,
                                       int dim_latent, Eigen::MatrixXd latent_initial,
                                       Eigen::VectorXd matrixBindices_,
                                       Eigen::VectorXd matrixBsubmeshes_,
                                       int total_mp_)
    : Nphi_u{dim_latent}, problem{&FOMproblem},
     matrixBindices{matrixBindices_}, total_mp{total_mp_},
     matrixBsubmeshes{matrixBsubmeshes_}
{

    // problem->L_Umodes is used to create volVectorFields
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 281 # " << endl;
    embedding_mdeim = autoPtr<EmbeddingMDEIM>(new EmbeddingMDEIM(Nphi_u, decoder_path, problem->L_Umodes[0], latent_initial));
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 283 # " << endl;
    // FOMproblem is only needed for initial conditions
    // TODO fix output dim
    newton_object_mdeim = newton_nmlspg_mdeim_burgers(Nphi_u, total_mp, *problem, embedding_mdeim.ref(), problem->L_Umodes[0]);

}

void NMLSPGMDEIMBurgers::NMLSPG_MMDEIM(int NmodesDEIMA, int NmodesDEIMB)
{
    // matrixMDEIMList is empty
    newton_object_mdeim.mdeim = autoPtr<DEIM_burgers>(new DEIM_burgers(matrixMDEIMList, NmodesDEIMA, NmodesDEIMB, "U_burgers"));
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 295 # " << endl;
    fvMesh &mesh = const_cast<fvMesh &>(problem->L_Umodes[0].mesh());
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 296 # " << endl;

    // Differential Operator
    newton_object_mdeim.mdeim->fieldsA = newton_object_mdeim.mdeim->generateSubmeshesMatrix(2, mesh, problem->L_Umodes[0]);
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 301 # " << endl;
    // Source Terms
    newton_object_mdeim.mdeim->fieldsB = newton_object_mdeim.mdeim->generateSubmeshesVector(2, mesh, problem->L_Umodes[0]);
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 304 # " << endl;

    newton_object_mdeim.matrixBindices = matrixBindices;
    newton_object_mdeim.total_mp = 0;

    // TODO fix
    // std::for_each(newton_object_mdeim.mdeim->submeshListA.begin(), newton_object_mdeim.mdeim->submeshListA.end(),
    // [&](fvMeshSubset
    // mesh_){newton_object_mdeim.total_mp+=mesh_.cellMap().size();});

    newton_object_mdeim.total_mp+= newton_object_mdeim.mdeim->magicPointsA.size();
    // for (int i = 0; i < newton_object_mdeim.mdeim->submeshListA.size(); i++)
    // {
    //     newton_object_mdeim.total_mp+=newton_object_mdeim.mdeim->submeshListA[i].cellMap().size();
    // }

    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 316 # " << newton_object_mdeim.total_mp << endl;
    std::cout << "indices B: " << matrixBindices.transpose() << std::endl;

    newton_object_mdeim.total_mp+=(matrixBindices.array()>0).count();
    // for (int i = 0; i < newton_object_mdeim.mdeim->submeshListB.size(); i++)
    // {
    //     newton_object_mdeim.total_mp+=newton_object_mdeim.mdeim->submeshListB[i].cellMap().size();
    // }
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 340 # total indices " << newton_object_mdeim.total_mp << endl;
    std::cout << "SUBMESHES: " << matrixBsubmeshes.transpose() << std::endl;
    newton_object_mdeim.total_submeshes_points = (matrixBsubmeshes.array()>=0).count();
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 506 # " << newton_object_mdeim.total_submeshes_points << endl;
    newton_object_mdeim.localAB2generalizedCoord(matrixBindices);
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 307 # " << endl;

    // Info << "The hyperreduced output has dimension: " << newton_object_mdeim.total_mp << endl;
    Eigen::VectorXd eigenU0 = Foam2Eigen::field2Eigen(problem->L_Umodes[0]);
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 308 # " << endl;
    newton_object_mdeim.U0_mdeim = newton_object_mdeim.projectVector(eigenU0);
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 309 # " << endl;
}

void NMLSPGMDEIMBurgers::solveOnline(Eigen::MatrixXd mu, int startSnap)
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
    // Info << " # DEBUG NMLSPGMDEIMBurgers_central.C, line 266 # " << endl;
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
    uRecLatent.resize((mu.cols()) * (onlineSizeTimeSeries), 2*newton_object_mdeim.total_submeshes_points);
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 360 # " << endl;
    // Iterate online solution for each parameter saved row-wise in mu
    for (int n_param = 0; n_param < mu.cols(); n_param++)
    {
        // std::cout << " # DEBUG NMLSPGMDEIMBurgers.C, line 364 # " << mu(0, n_param) << std::endl;
        // Set the initial time
        time = tstart;

        // Counter of the number of saved time steps for the present parameter with index n_param
        int counter = 0;
        int nextStore = 0;

        // residual export for hyper-reduction counter
        int counterResidual = 1;
        int counterMatrixMDEIM = 1;

        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 376 # " << endl;
        // Create and resize the solution vector (column vector)
        y.resize(Nphi_u, 1);
        y = embedding_mdeim->latent_initial.transpose();
        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 380 # " << endl;
        auto refTerm = newton_object_mdeim.embedding_ref(mu(0, n_param));
        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 382 # " << endl;
        newton_object_mdeim.g_old = embedding_mdeim->forward(y, mu(0, n_param)) + refTerm ;
        newton_object_mdeim.init_old(newton_object_mdeim.g_old);

        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 381 # " << endl;
        // Set some properties of the newton object
        newton_object_mdeim.mu = mu(0, n_param);
        newton_object_mdeim.nu = nu;
        newton_object_mdeim.dt = dt;
        newton_object_mdeim.tauU = tauU;

        // Create vector to store temporal solution and save initial condition
        // as first solution
        Eigen::MatrixXd tmp_sol(Nphi_u + 2, 1);
        tmp_sol(0) = time;
        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 397 # " << endl;
        tmp_sol(1) = newton_object_mdeim.mu;
        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 399 # " << endl;
        tmp_sol.col(0).tail(y.rows()) = y;
        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 401 # " << endl;
        online_solution[counter2] = tmp_sol;
        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 403 # " << endl;
        uRecLatent.row(counter2) = newton_object_mdeim.g_old;
        counter2++;
        counter++;
        nextStore += numberOfStores;

        // Create nonlinear solver object
        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 401 # " << endl;
        // Create nonlinear solver object
        Eigen::NumericalDiff<newton_nmlspg_mdeim_burgers, Eigen::Central> numDiffobject(newton_object_mdeim, 1.e-01);
        Eigen::LevenbergMarquardt<decltype(numDiffobject)> lm(numDiffobject);

        // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 405 # " << endl;
        lm.parameters.factor = 100;       //step bound for the diagonal shift, is this related to damping parameter, lambda?
        lm.parameters.maxfev = 5000;      //max number of function evaluations
        lm.parameters.xtol = 1.49012e-20; //tolerance for the norm of the solution vector
        lm.parameters.ftol = 1.49012e-20; //tolerance for the norm of the vector function
        lm.parameters.gtol = 0;           // tolerance for the norm of the gradient of the error vector
        lm.parameters.epsfcn = 0;         //error precision

        // Set output colors for fancy output
        Color::Modifier red(Color::FG_RED);
        Color::Modifier green(Color::FG_GREEN);
        Color::Modifier def(Color::FG_DEFAULT);

        time = time + dt;

        while (time < finalTime)
        {
            // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 422 # " << endl;
            auto start = std::chrono::system_clock::now();
            Eigen::LevenbergMarquardtSpace::Status ret = lm.minimize(y);
            auto end = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            std::cout << "LM finished with status: " << ret << std::endl;
            std::cout << " minimum: " << y.transpose() << endl;

            // evaluate residual for postProcessing and Info
            Eigen::VectorXd res(newton_object_mdeim.total_mp);
            res.setZero();

            numDiffobject(y, res);

            // update the old solution for the evaluation of the residual
            numDiffobject.g_old = embedding_mdeim->forward(y, mu(0, n_param)) + numDiffobject.embedding_ref(mu(0, n_param));
            // update the List of submeshes fields
            numDiffobject.init_old(numDiffobject.g_old);
            // DEBUG utility: reconstruct residual
            numDiffobject.rec_field(res, problem->L_Umodes[0], "res");
            // DEBUG utility: reconstruct compressed decoder output on
            // magicPoints only
            auto vec_rec = numDiffobject.restrict_decoder();
            numDiffobject.rec_field(vec_rec, problem->L_Umodes[0], "U");


            std::cout << "################## Online solve N° " << counter << " ##################" << std::endl;
            // Info << "Time = " << time << endl;

            if (res.norm() < 1e-5)
            {
                std::cout << green << "|F(x)| = " << res.norm() << " - Minimun reached in " << lm.iter << " iterations " << def << std::endl;
            }
            else
            {
                std::cout << red << "|F(x)| = " << res.norm() << " - Minimun reached in " << lm.iter << " iterations " << def << std::endl;
            }
            std::cout << "Elapsed in: " << elapsed.count() << " microseconds" <<std::endl << std::endl;

            tmp_sol(0) = time;
            tmp_sol(1) = newton_object_mdeim.mu;
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
                    uRecLatent.row(counter2) = numDiffobject.g_old;
                }

                nextStore += numberOfStores;
                counter2++;
            }

            counter++;
            time = time + dt;
        }
    }
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 495 # " << endl;
    cnpy::save(uRecLatent, "./ITHACAoutput/DEIM/latentHyperReduced.npy");
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 497 # " << endl;
}

// * * * * * * * * * * * * * * *  Evaluation  * * * * * * * * * * * * * //

void NMLSPGMDEIMBurgers::reconstruct(fileName decoder_path,
                                     Burgers& problem, bool exportFields,
                                     fileName folder)
{
    if (exportFields)
    {
        mkDir(folder);
        ITHACAutilities::createSymLink(folder);
    }

    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 516 # " << endl;
    // problem->L_Umodes is used to create volVectorFields
    embedding = autoPtr<Embedding>(new Embedding(Nphi_u, decoder_path, problem.L_Umodes[0], embedding_mdeim->latent_initial));
    // Info << " # DEBUG NMLSPGMDEIMBurgers.C, line 519 # " << endl;

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
            // currentUCoeff = online_solution[i].block(1, 0, Nphi_u, 1);
            currentUCoeff = online_solution[i].col(0).tail(y.rows());
            CoeffU.append(currentUCoeff);
            nextwrite += exportEveryIndex;
            double timeNow = online_solution[i](0);
            tValues.append(timeNow);
            double mu = online_solution[i](1);

            auto tmp = embedding->forward(currentUCoeff, mu);
            uRecFields.append(tmp().clone());
        }

        counter++;
    }

    if (exportFields)
    {
        ITHACAstream::exportFields(uRecFields, folder,
                                   "uRec");
    }
}


Eigen::MatrixXd NMLSPGMDEIMBurgers::setOnlineVelocity(Eigen::MatrixXd vel)
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

void NMLSPGMDEIMBurgers::trueProjection(fileName folder)
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
