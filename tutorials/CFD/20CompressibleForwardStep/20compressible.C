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
Description
    Example of a Burgers' Problem
SourceFiles
    00burgers.C
\*---------------------------------------------------------------------------*/

#include <torch/script.h>
#include <torch/torch.h>
#include "torch2Eigen.H"
#include "torch2Foam.H"
#include "Foam2Eigen.H"

#include "compressibleNS.H"
#include "ITHACAPOD.H"
#include "DEIM.H"
#include "ITHACAstream.H"
#include "cnpy.H"
#include <chrono>
#include <math.h>
#include <iomanip>
#include <string>
#include <algorithm>

#include "ConvAe.H"

class tutorial00 : public compressibleNS
{
public:
    PtrList<volScalarField> U0field;
    PtrList<volScalarField> U1field;

    PtrList<volScalarField> U0modes;
    PtrList<volScalarField> U1modes;

    explicit tutorial00(int argc, char *argv[])
        : compressibleNS(argc, argv),
          U(_U())
    {
    }

    // Fields for parameter dependence
    volVectorField &U;

    void offlineSolveMach(fileName folder = "./ITHACAoutput/Offline/")
    {
        List<scalar> mu_now(1);

        if (offline)
        {
            ITHACAstream::read_fields(Ufield, "U", folder, 1);
            ITHACAstream::read_fields(rhofield, "rho", folder, 1);
            ITHACAstream::read_fields(efield, "e", folder, 1);
        }
        else
        {
            std::cout << "Current mu = " << mu(0, 0) << std::endl;
            mu_now[0] = mu(0, 0);
            change_initial_velocity(mu(0, 0));
            truthSolve(mu_now, folder);

            for (label i = 1; i < mu.cols(); i++)
            {
                std::cout << "Current mu = " << mu(0, i) << std::endl;
                mu_now[0] = mu(0, i);
                restart();
                change_initial_velocity(mu(0, i));
                truthSolve(mu_now, folder);
            }
        }
    }

    // ! it would be more efficient to implement inside getModesRSVD
    torch::Tensor compress(PtrList<volScalarField>& fields, PtrList<volScalarField>& modes, fileName name)
    {
        Info << "Compress field "  << endl;

        Eigen::MatrixXd compressed;

        Eigen::MatrixXd SnapMatrix = Foam2Eigen::PtrList2Eigen(fields);
        Info << "Snap shapes: " << SnapMatrix.rows() << " x " <<SnapMatrix.cols() << endl;

        Eigen::MatrixXd Modes = Foam2Eigen::PtrList2Eigen(modes);
        Info << "Modes shapes: " << Modes.rows() << " x " << Modes.cols() << endl;

        Eigen::MatrixXd modesTransposed = Modes.transpose();
        torch::Tensor modesTransposedTorch = ITHACAtorch::torch2Eigen::eigenMatrix2torchTensor(modesTransposed);
        torch::save(modesTransposedTorch, "modes" + name + ".pt");

        compressed = modesTransposed * SnapMatrix;

        torch::Tensor tensor = ITHACAtorch::torch2Eigen::eigenMatrix2torchTensor(compressed);


        std::cout << "End compress field with size " << tensor.sizes() << std::endl;
        return tensor;
    };

    /// Compress 2d vectorial fields
//     torch::Tensor compress(PtrList<volVectorField>& fields, PtrList<volVectorField>& modes, fileName name)
//     {
//         Info << "Compress fields "  << endl;

//         Eigen::MatrixXd snapMatrix = Foam2Eigen::PtrList2Eigen(fields);
//         Info << "Snap shapes: " <<snapMatrix.rows() << " x " <<snapMatrix.cols() << endl;

//         Eigen::MatrixXd Modes = Foam2Eigen::PtrList2Eigen(modes);
//         Info << "Modes shapes: " << Modes.rows() << " x " << Modes.cols() << endl;

//         // dimension of a single component of a vectorial snapshot
//         int dofScalar = Modes.rows()/3;
//         std::cout << "dofScalar: " << dofScalar << std::endl;

//         // first component
//         Eigen::MatrixXd compressed1;
//         Eigen::MatrixXd modesTransposed1 = Modes.block(0, 0, dofScalar, Modes.cols()).transpose();
//         torch::Tensor modesTransposedTorch1 = ITHACAtorch::torch2Eigen::eigenMatrix2torchTensor(modesTransposed1);
//         torch::save(modesTransposedTorch1, "modes" + name + "_0.pt");

//         compressed1 = modesTransposed1 * snapMatrix.block(0, 0, dofScalar, snapMatrix.cols());
// //         cnpy::save(compressed1, name + "_0.npy");
//         // cnpy::save(modesTransposed1, "modes" + name + "_0.npy");

//         std::cout << "Compressed field first component " << compressed1.rows() << " x " << compressed1.cols() << std::endl;
//         std::cout << "Compressed modes first component " << modesTransposed1.rows() << " x " << modesTransposed1.cols() << std::endl;

//         // second component
//         Eigen::MatrixXd compressed2;
//         Eigen::MatrixXd modesTransposed2 = Modes.block(dofScalar, 0, dofScalar, Modes.cols()).transpose();
//         torch::Tensor modesTransposedTorch2 = ITHACAtorch::torch2Eigen::eigenMatrix2torchTensor(modesTransposed2);
//         torch::save(modesTransposedTorch2, "modes" + name + "_1.pt");

//         compressed2 = modesTransposed2 * snapMatrix.block(dofScalar, 0, dofScalar, snapMatrix.cols());
//         // cnpy::save(compressed2, name + "_1.npy");
//         // cnpy::save(modesTransposed2, "modes" + name + "_1.npy");

//         std::cout << "Compressed field second component " << compressed2.rows() << " x " << compressed2.cols() << std::endl;
//         std::cout << "Compressed modes second component " << modesTransposed2.rows() << " x " << modesTransposed2.cols() << std::endl;

//         torch::Tensor tensor1 = ITHACAtorch::torch2Eigen::eigenMatrix2torchTensor(compressed1);
//         torch::Tensor tensor2 = ITHACAtorch::torch2Eigen::eigenMatrix2torchTensor(compressed2);
//         auto tensor = torch::cat({tensor1, tensor2}, 0);

//         std::cout << "End compress field with size " << tensor.sizes() << std::endl;
//         return tensor;
//     };
};

/*---------------------------------------------------------------------------*\
                               Starting the MAIN
\*---------------------------------------------------------------------------*/

void one_parameter_viscosity(tutorial00);
void train_one_parameter_initial_velocity(tutorial00);
void test_one_parameter_initial_velocity(tutorial00);
void nonlinear_one_parameter_initial_velocity(tutorial00);
void nonlinear_test_rec(tutorial00);
void nonlinear_test_data(tutorial00);
void nonlinear_one_parameter_initial_velocity_hr(tutorial00);

void train_ConvAe(tutorial00);
void decompress(tutorial00);

int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        std::cout << "Pass train or test." << endl;
        exit(0);
    }
    // processed arguments
    int argc_proc = argc - 1;
    char *argv_proc[argc_proc];
    argv_proc[0] = argv[0];

    if (argc > 2)
    {
        std::copy(argv + 2, argv + argc, argv_proc + 1);
    }

    tutorial00 example(argc_proc, argv_proc);

    if (std::strcmp(argv[1], "train") == 0)
    {
        // save mu_samples and training snapshots reduced coefficients
        train_one_parameter_initial_velocity(example);
    }
    else if (std::strcmp(argv[1], "ae") == 0)
    {
        train_ConvAe(example);
    }
    // else if (std::strcmp(argv[1], "decompress") == 0)
    // {
    //     decompress(example);
    // }
    else if (std::strcmp(argv[1], "test") == 0)
    {
        // compute FOM, ROM-intrusive, ROM-nonintrusive and evaluate errors
        test_one_parameter_initial_velocity(example);
    }
    // else if (std::strcmp(argv[1], "nonlinear") == 0)
    // {
    //     // compute NM-LSPG and evaluate errors
    //     nonlinear_one_parameter_initial_velocity(example);
    // }
    // else if (std::strcmp(argv[1], "nltestrec") == 0)
    // {
    //     // reconstruct NM projected solutions and compute NM projection error
    //     nonlinear_test_rec(example);
    // }
    // else if (std::strcmp(argv[1], "nltestdata") == 0)
    // {
    //     // reconstruct NM-LSTM predicted solutions and compute rel L2 error
    //     nonlinear_test_data(example);
    // }
    // else if (std::strcmp(argv[1], "hr") == 0)
    // {
    //     // NM-LSPG-HR
    //     nonlinear_one_parameter_initial_velocity_hr(example);
    // }
    else
    {
        std::cout << "Pass train or test." << std::endl;
    }

    exit(0);
}

void train_one_parameter_initial_velocity(tutorial00 train_FOM)
{
    // Read parameters from ITHACAdict file
    ITHACAparameters *para = ITHACAparameters::getInstance(train_FOM._mesh(),
                                                           train_FOM._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);

    /// Set the number of parameters
    train_FOM.Pnumber = 1;
    /// Set the dimension of the training set
    train_FOM.Tnumber = 10;
    /// Instantiates a void Pnumber-by-Tnumber matrix mu for the parameters and a void
    /// Pnumber-by-2 matrix mu_range for the ranges
    train_FOM.setParameters();
    // Set the parameter ranges
    train_FOM.mu_range(0, 0) = 3.3;
    train_FOM.mu_range(0, 1) = 3.9;
    // Generate a number of Tnumber linearly equispaced samples inside the parameter range
    train_FOM.genEquiPar();
    cnpy::save(train_FOM.mu, "parTrain.npy");

    // Time parameters
    train_FOM.startTime = 0;
    train_FOM.finalTime = 10;
    train_FOM.timeStep = 0.002;
    train_FOM.writeEvery = 0.01;

    // Perform The Offline Solve;
    train_FOM.offlineSolveMach("./ITHACAoutput/Offline/Training/");

    // Perform the compression of the U, rho, e fields with rSVD
    int m{6}; // m x m frame dimension
    int NmodesCompression{pow(2, 2*m)};

    // Perform a POD wiht rSVD decomposition for velocity
    Info << endl << "Start compressing with reduced dimension " << NmodesCompression << endl;

    // Get component lists

    for (int i = 0; i < train_FOM.Ufield.size(); i++)
    {
        train_FOM.U0field.append(train_FOM.Ufield[i].component(0));
        train_FOM.U1field.append(train_FOM.Ufield[i].component(1));
    }

    // ITHACAstream::read_fields(train_FOM.U0modes, "U.component(0)", "./ITHACAoutput/POD/", 1);
    ITHACAPOD::getModesSVD(train_FOM.U0field, train_FOM.U0modes, train_FOM.U0field[0].name(), 0, 0, 0, NmodesCompression);

    ITHACAstream::read_fields(train_FOM.U1modes, train_FOM.U1field[0].name(), "./ITHACAoutput/POD/", 1);
    // ITHACAPOD::getModesSVD(train_FOM.U1field, train_FOM.U1modes, "U_1",0, 0, 0, NmodesCompression);

    ITHACAstream::read_fields(train_FOM.rhomodes, "rho", "./ITHACAoutput/POD/", 1);
    // ITHACAPOD::getModesSVD(train_FOM.rhofield, train_FOM.rhomodes, "rho",/*train_FOM.podex*/ 0, 0, 0, NmodesCompression);

    ITHACAstream::read_fields(train_FOM.emodes, "e", "./ITHACAoutput/POD/", 1);
    // ITHACAPOD::getModesSVD(train_FOM.efield, train_FOM.emodes, "e", /*train_FOM.podex*/ 0, 0, 0, NmodesCompression);

    torch::Tensor U0compressed;
    torch::Tensor U1compressed;
    torch::Tensor rhocompressed;
    torch::Tensor ecompressed;

    int n_snap = train_FOM.efield.size();
    U0compressed = train_FOM.compress(train_FOM.U0field, train_FOM.U0modes, "U_0").transpose(0, 1).reshape({n_snap, 1, pow(2, m), pow(2, m)}).contiguous();
    U1compressed = train_FOM.compress(train_FOM.U1field, train_FOM.U1modes, "U_1").transpose(0, 1).reshape({n_snap, 1, pow(2, m), pow(2, m)}).contiguous();
    rhocompressed = train_FOM.compress(train_FOM.rhofield, train_FOM.rhomodes, "rho").transpose(0, 1).reshape({n_snap, 1, pow(2, m), pow(2, m)}).contiguous();
    ecompressed = train_FOM.compress(train_FOM.efield, train_FOM.emodes, "e").transpose(0, 1).reshape({n_snap, 1, pow(2, m), pow(2, m)}).contiguous();

    // torch::save(Ucompressed, "./ITHACAoutput/compressed/compressedU.pt");
    // torch::save(rhocompressed, "./ITHACAoutput/compressed/compressedrho.pt");
    // torch::save(ecompressed, "./ITHACAoutput/compressed/compressede.pt");
    // Info << " Torch tensors compressed saved " << endl;

    // stack w.r.t. r the dimension reduced with rSVD : r x n_snap
    torch::Tensor tensor = torch::cat({std::move(U0compressed), std::move(U1compressed), std::move(rhocompressed), std::move(ecompressed)}, 1);
    std::cout << "Compressed tensor shape: " << tensor.sizes() << std::endl;
    torch::save(tensor, "compressedSnap.pt");
    tensor = tensor.reshape({n_snap, 4*pow(2, m)*pow(2, m)});
    Eigen::MatrixXd tensorEig = ITHACAtorch::torch2Eigen::torchTensor2eigenMatrix<double>(tensor);
    cnpy::save(tensorEig, "compressedSnap.npy");
}

void test_one_parameter_initial_velocity(tutorial00 test_FOM)
{
    // Read parameters from ITHACAdict file
    ITHACAparameters *para = ITHACAparameters::getInstance(test_FOM._mesh(),
                                                           test_FOM._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);

    /// Set the number of parameters
    test_FOM.Pnumber = 1;
    /// Set the dimension of the test set
    test_FOM.Tnumber = 1;
    // sample test set
    test_FOM.setParameters();
    // Set the parameter ranges
    test_FOM.mu_range(0, 0) = 3.6;
    test_FOM.mu_range(0, 1) = 3.6;
    // Generate a number of Tnumber linearly equispaced samples inside the parameter range
    test_FOM.genEquiPar();
    cnpy::save(test_FOM.mu, "parTest.npy");

    // Generate a number of Tnumber linearly equispaced samples inside the parameter range
    // Eigen::MatrixXd mu;
    // test_FOM.mu = cnpy::load(mu, "parTest.npy");

    // Time parameters
    test_FOM.startTime = 0;
    test_FOM.finalTime = 10;
    test_FOM.timeStep = 0.002;
    test_FOM.writeEvery = 0.01;

    // Perform The Offline Solve;
    if (!ITHACAutilities::check_folder("./ITHACAoutput/Offline/Test/"))
    {
        test_FOM.offline = false;
    }

    test_FOM.offlineSolveMach("./ITHACAoutput/Offline/Test/");
    Eigen::MatrixXd trueSnapMatrix = Foam2Eigen::PtrList2Eigen(test_FOM.Ufield);
    // cnpy::save(trueSnapMatrix, "npTrueSnapshots.npy");

    int m{6}; // m x m frame dimension
    int NmodesCompression{pow(2, 2*m)};

    Info << endl << "Start compressing with reduced dimension " << NmodesCompression << endl;

    ITHACAstream::read_fields(test_FOM.U0modes, "U.component(0)", "./ITHACAoutput/POD/", 1);
    ITHACAstream::read_fields(test_FOM.U1modes, "U.component(1)", "./ITHACAoutput/POD/", 1);
    ITHACAstream::read_fields(test_FOM.rhomodes, "rho", "./ITHACAoutput/POD/", 1);
    ITHACAstream::read_fields(test_FOM.emodes, "e", "./ITHACAoutput/POD/", 1);

    torch::Tensor U0compressed;
    torch::Tensor U1compressed;
    torch::Tensor rhocompressed;
    torch::Tensor ecompressed;

    int n_snap = test_FOM.efield.size();
    for (int i = 0; i < test_FOM.Ufield.size(); i++)
    {
        test_FOM.U0field.append(test_FOM.Ufield[i].component(0));
        test_FOM.U1field.append(test_FOM.Ufield[i].component(1));
    }

    Info << " # DEBUG 20compressible.C, line 230 # " << n_snap << endl;
    U0compressed = test_FOM.compress(test_FOM.U0field, test_FOM.U0modes, "U_0").transpose(0, 1).reshape({n_snap, 1, pow(2, m), pow(2, m)}).contiguous();
    U1compressed = test_FOM.compress(test_FOM.U1field, test_FOM.U1modes, "U_1").transpose(0, 1).reshape({n_snap, 1, pow(2, m), pow(2, m)}).contiguous();
    rhocompressed = test_FOM.compress(test_FOM.rhofield, test_FOM.rhomodes, "rho").transpose(0, 1).reshape({n_snap, 1, pow(2, m), pow(2, m)}).contiguous();
    ecompressed = test_FOM.compress(test_FOM.efield, test_FOM.emodes, "e").transpose(0, 1).reshape({n_snap, 1, pow(2, m), pow(2, m)}).contiguous();

    // stack w.r.t. r the dimension reduced with rSVD : r x n_snap
    torch::Tensor tensor = torch::cat({std::move(U0compressed), std::move(U1compressed), std::move(rhocompressed), std::move(ecompressed)}, 1);

    std::cout << " # DEBUG 20compressible.C, line 238 # tensor shape: " << tensor.sizes() << std::endl;

    torch::save(tensor, "compressedSnapTest.pt");
    tensor = tensor.reshape({n_snap, 4*pow(2, m)*pow(2, m)});
    Eigen::MatrixXd tensorEig =
    ITHACAtorch::torch2Eigen::torchTensor2eigenMatrix<double>(tensor);
    cnpy::save(tensorEig, "compressedSnapTest.npy");
    tensor = tensor.reshape({n_snap, 4, pow(2, m)*pow(2, m)});

    // Decompress rho test
    torch::Tensor modesrho;
    torch::load(modesrho, "modesrho.pt");
    std::cout << "mode rho " << modesrho.sizes() << std::endl;

    torch::Tensor rho = tensor.index({torch::indexing::Slice(), 2, torch::indexing::Slice()});
    std::cout << "rho: " << rho.sizes() << std::endl;

    torch::Tensor rho_dec = torch::matmul(rho, modesrho);
    std::cout << "rho_dec: " << rho_dec.sizes() << std::endl;

    auto rho_list = ITHACAtorch::torch2Foam::torch2PtrList<scalar>(rho_dec);
    auto field_rho =  test_FOM.rhofield[0];

    PtrList<volScalarField> rho_list_;

    for(int i=0; i<rho_list.size(); i++)
    {
        field_rho.ref().field() = rho_list[i];
        rho_list_.append(field_rho);
    }

    ITHACAstream::exportFields(rho_list_, "rho_decompressed", "rho");

    // Decompress e test
    torch::Tensor modese;
    torch::load(modese, "modese.pt");
    std::cout << "mode e " << modese.sizes() << std::endl;

    torch::Tensor e = tensor.index({torch::indexing::Slice(), 3, torch::indexing::Slice()});
    std::cout << "e: " << e.sizes() << std::endl;

    torch::Tensor e_dec = torch::matmul(e, modese);
    std::cout << "e_dec: " << e_dec.sizes() << std::endl;

    auto e_list = ITHACAtorch::torch2Foam::torch2PtrList<scalar>(e_dec);
    auto field_e =  test_FOM.efield[0];

    PtrList<volScalarField> e_list_;

    for(int i=0; i<e_list.size(); i++)
    {
        field_e.ref().field() = e_list[i];
        e_list_.append(field_e);
    }

    ITHACAstream::exportFields(e_list_, "e_decompressed", "e");

    // Decompress U test
    torch::Tensor modesU_0;
    torch::load(modesU_0, "modesU_0.pt");
    std::cout << "mode U0 " << modesU_0.sizes() << std::endl;

    torch::Tensor U_0 = tensor.index({torch::indexing::Slice(), 0, torch::indexing::Slice()});
    std::cout << "U_0: " << U_0.sizes() << std::endl;

    torch::Tensor modesU_1;
    torch::load(modesU_1, "modesU_1.pt");
    std::cout << "mode U1 " << modesU_1.sizes() << std::endl;

    torch::Tensor U_1 = tensor.index({torch::indexing::Slice(), 1, torch::indexing::Slice()});
    std::cout << "U_1: " << U_1.sizes() << std::endl;

    torch::Tensor U_0_dec = torch::matmul(U_0, modesU_0).unsqueeze(2);
    std::cout << "U_0_dec: " << U_0_dec.sizes() << std::endl;

    torch::Tensor U_1_dec = torch::matmul(U_1, modesU_1).unsqueeze(2);
    std::cout << "U_1_dec: " << U_1_dec.sizes() << std::endl;

    torch::Tensor U_dec = torch::cat({U_0_dec, U_1_dec, torch::zeros_like(U_0_dec)}, 2).view({n_snap, -1}).contiguous();
    std::cout << "U_dec: " << U_dec.sizes() << std::endl;

    auto U_list = ITHACAtorch::torch2Foam::torch2PtrList<vector>(U_dec);
    auto field_U =  test_FOM.Ufield[0];

    PtrList<volVectorField> U_list_;

    for(int i=0; i<U_list.size(); i++)
    {
        field_U.ref().field() = U_list[i];
        U_list_.append(field_U);
    }

    ITHACAstream::exportFields(U_list_, "U_decompressed", "U");

    Info << endl << "Evaluate fields projection error ..." << endl;
    // Compression error U
    Eigen::MatrixXd errL2U = ITHACAutilities::errorL2Rel(test_FOM.Ufield, U_list_);
    // ITHACAstream::exportMatrix(errL2U, "errL2U", "matlab","./ITHACAoutput/ErrorsL2/");
    // cnpy::save(errL2U, "./ITHACAoutput/ErrorsL2/errL2U.npy");
    std::cout << "Mean U L2 compression error: " << errL2U.mean() << std::endl;
    // Compression error rho
    Eigen::MatrixXd errL2rho = ITHACAutilities::errorL2Rel(test_FOM.rhofield, rho_list_);
    // ITHACAstream::exportMatrix(errL2rho, "errL2rho", "matlab","./ITHACAoutput/ErrorsL2/");
    // cnpy::save(errL2rho, "./ITHACAoutput/ErrorsL2/errL2rho.npy");
    std::cout << "Mean rho L2 compression error: " << errL2rho.mean() << std::endl;

    // Compression error e
    Eigen::MatrixXd errL2e = ITHACAutilities::errorL2Rel(test_FOM.efield, e_list_);
    // ITHACAstream::exportMatrix(errL2e, "errL2e", "matlab","./ITHACAoutput/ErrorsL2/");
    // cnpy::save(errL2e, "./ITHACAoutput/ErrorsL2/errL2e.npy");
    std::cout << "Mean e L2 compression error: " << errL2e.mean() << std::endl;

    // Operator projection error
    Info << endl << "Evaluate operator projection error ..." << endl;
    int dof = modesU_0.sizes()[1];
    torch::Tensor projErrorU_0 = torch::norm(torch::eye(dof)-torch::matmul(modesU_0.transpose(1, 0), modesU_0));
    std::cout << "Projection error U_0 " << projErrorU_0 << std::endl;
    std::cout << torch::matmul(modesU_0.transpose(1, 0), modesU_0).index({torch::indexing::Slice(0, 3, 1), torch::indexing::Slice(0, 3, 1)}) << std::endl;
    std::cout << "Norm : " << torch::norm(torch::slice(modesU_0, 1, 0, dof, 1)) << std::endl;

    torch::Tensor projErrorU_1 = torch::norm(torch::eye(dof)-torch::matmul(modesU_1.transpose(1, 0), modesU_1));
    std::cout << "Projection error U_1 " << projErrorU_1 << std::endl;

    torch::Tensor projErrorrho = torch::norm(torch::eye(dof)-torch::matmul(modesrho.transpose(1, 0), modesrho));
    std::cout << "Projection error rho " << projErrorrho << std::endl;

    torch::Tensor projErrore = torch::norm(torch::eye(dof)-torch::matmul(modese.transpose(1, 0), modese));
    std::cout << "Projection error e " << projErrore << std::endl;

}

void train_ConvAe(tutorial00 train_FOM)
{
    /// Load train data
    // Eigen::MatrixXd compressedEig;
    // compressedEig = cnpy::load(compressedEig, "compressedSnap.npy");

    torch::Tensor compressedTorch;
    torch::load(compressedTorch, "compressedSnap.pt");//ITHACAtorch::torch2Eigen::eigenMatrix2torchTensor(compressedEig);
    int m{6};
    int n_train = compressedTorch.sizes()[0];
    // compressedTorch = compressedTorch.view({-1, 4, pow(2, m), pow(2, m)});
    std::cout << "Loaded train dataset with shape " << compressedTorch.sizes() << std::endl;

    // initialize normalization parameters
    torch::Tensor min_sn = std::get<0>(torch::min(compressedTorch.transpose(1, 0).reshape({4, 5005*pow(64, 2)}), 1));
    torch::Tensor max_sn = std::get<0>(torch::max(compressedTorch.transpose(1, 0).reshape({4, 5005*pow(64, 2)}), 1));
    std::cout << "Min values: " << min_sn.view({1, -1}) << std::endl;
    std::cout << "Max values: " << max_sn.view({1, -1}) << std::endl;

    torch::Tensor scaledTorch = (compressedTorch - min_sn.view({1, -1, 1, 1}))/(max_sn-min_sn).view({1, -1, 1, 1});

    torch::Tensor min_sn_ = std::get<0>(torch::min(scaledTorch.transpose(1, 0).reshape({4, 5005*pow(64, 2)}), 1));
    torch::Tensor max_sn_ = std::get<0>(torch::max(scaledTorch.transpose(1, 0).reshape({4, 5005*pow(64, 2)}), 1));
    std::cout << "Min values: " << min_sn_.view({1, -1}) << std::endl;
    std::cout << "Max values: " << max_sn_.view({1, -1}) << std::endl;

    // save normalizing tensors
    torch::save(min_sn , "min_sn.pt");
    torch::save(max_sn , "max_sn.pt");
    std::cout << "Saved normalizing tensors" << std::endl;

    auto data_set = autoPtr<SnapDataset>(new SnapDataset(scaledTorch));

    /// Load test data
    torch::Tensor compressedTorchTest;
    torch::load(compressedTorchTest, "compressedSnapTest.pt");
    std::cout << "Loaded test dataset with shape " << compressedTorchTest.sizes() << std::endl;
    int n_test = compressedTorchTest.sizes()[0];

    torch::Tensor min_sn_test = std::get<0>(torch::min(compressedTorchTest.transpose(1, 0).reshape({4, n_test*pow(64, 2)}), 1));
    torch::Tensor max_sn_test = std::get<0>(torch::max(compressedTorchTest.transpose(1, 0).reshape({4, n_test*pow(64, 2)}), 1));
    std::cout << "Min values test: " << min_sn_test.view({1, -1}) << std::endl;
    std::cout << "Max values test: " << max_sn_test.view({1, -1}) << std::endl;

    torch::Tensor scaledTorchTest = (compressedTorchTest - min_sn.view({1, -1, 1, 1}))/(max_sn-min_sn).view({1, -1, 1, 1});

    min_sn_test = std::get<0>(torch::min(scaledTorchTest.transpose(1, 0).reshape({4, n_test*pow(64, 2)}), 1));
    max_sn_test = std::get<0>(torch::max(scaledTorchTest.transpose(1, 0).reshape({4, n_test*pow(64, 2)}), 1));
    std::cout << "Min values test: " << min_sn_test.view({1, -1}) << std::endl;
    std::cout << "Max values test: " << max_sn_test.view({1, -1}) << std::endl;

    /// Train

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper-parameters
    const int64_t batch_size = 100;
    const size_t num_epochs = 100;
    const double learning_rate = 1e-3;
    const int testEval = 10;

    // Autoencoder training parameters
    auto autoencoder = std::shared_ptr<Autoencoder>(new Autoencoder(4));
    autoencoder->to(device);

    auto optimizer = new torch::optim::Adam(autoencoder->parameters(),
                                               torch::optim::AdamOptions(learning_rate));
    // Generate a data loader.
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    std::move(data_set()),
    batch_size);

    torch::Tensor snap_rec;

    if (!ITHACAutilities::check_file("convae.pt"))
    {
        std::cout << "Training...\n";
        autoencoder->train();
        auto train_norm = torch::frobenius_norm(scaledTorch);

        for (int64_t epoch = 1; epoch <= num_epochs; ++epoch)
        {
            torch::Tensor loss{torch::zeros({1})};
            loss = loss.to(device);

            for (auto& batch : *data_loader)
            {
                optimizer->zero_grad();

                torch::Tensor data = torch::zeros({batch_size, 4, pow(2, m), pow(2, m)}) ;
                torch::Tensor bb;

                // TODO fix the batch initialization
                for (int i; i< batch_size; i++)
                {
                    bb = batch[i];
                    // std::cout << "bb " << bb.size(1) << std::endl;
                    data.slice(0, i, i+1)=batch[i];
                }

                // std::cout << "batch shape " << data.sizes() << std::endl;

                torch::Tensor snap_rec = autoencoder->forward(data.to(at::kFloat).to(device));

                // std::cout << "rec shape " << snap_rec.size(0) << " " << snap_rec.size(1) << std::endl;

                auto loss_ = torch::nn::functional::mse_loss(snap_rec, data.to(device), torch::kSum);

                loss_.backward();
                optimizer->step();

                // std::cout << loss_.item<float>() << std::endl;
                loss = loss + loss_;
            }

            auto mean_loss = loss / n_train;

            std::cout << "Epoch : " << epoch << "/" <<  num_epochs << " Mean RMS: " << mean_loss.item<float>() << std::endl;

            if (epoch % testEval == 0)
            {
                autoencoder->to(torch::kCPU);
                snap_rec = autoencoder->forward(scaledTorchTest.to(at::kFloat).to(torch::kCPU));
                // auto train_rel_loss =
                // torch::frobenius_norm(snap_rec-scaledTorchTest)/train_norm;
                auto test_rel_loss = torch::max(torch::abs(snap_rec-scaledTorchTest));
                std::cout << "Train, max Rel Loss: " << test_rel_loss.item<float>() << std::endl;
                autoencoder->to(device);
            }

        }

        // save net
        torch::save(autoencoder, "convae.pt");
        torch::save(snap_rec, "snap_rec.pt");
        std::cout << std::endl << "Net saved" << std::endl;
    }

    // save net
    torch::load(autoencoder, "convae.pt");
    torch::load(snap_rec, "snap_rec.pt");

    Info << endl << "Load modes ..." << endl;
    torch::Tensor modesU0;
    torch::load(modesU0, "modesU_0.pt");
    std::cout << "mode U0 " << modesU0.sizes() << std::endl;

    torch::Tensor modesU1;
    torch::load(modesU1, "modesU_1.pt");
    std::cout << "mode U1 " << modesU1.sizes() << std::endl;

    torch::Tensor modesrho;
    torch::load(modesrho, "modesrho.pt");
    std::cout << "mode rho " << modesrho.sizes() << std::endl;

    torch::Tensor modese;
    torch::load(modese, "modese.pt");
    std::cout << "mode e " << modese.sizes() << std::endl;

    // Decompress
    Info << endl << "Load compressed fields ..." << endl;
    snap_rec = snap_rec * (max_sn-min_sn).view({1, -1, 1, 1}) + min_sn.view({1, -1, 1, 1});
    snap_rec = snap_rec.view({-1, 4, pow(2, 2*m)});
    std::cout << "snap_rec: " << snap_rec.sizes() << std::endl;

    torch::Tensor U0 = snap_rec.index({torch::indexing::Slice(), 0, torch::indexing::Slice()});
    std::cout << "U0: " << U0.sizes() << std::endl;
    Eigen::MatrixXd U0Eig = ITHACAtorch::torch2Eigen::torchTensor2eigenMatrix<double>(U0);

    torch::Tensor U1 = snap_rec.index({torch::indexing::Slice(), 1, torch::indexing::Slice()});
    std::cout << "U1: " << U1.sizes() << std::endl;
    Eigen::MatrixXd U1Eig = ITHACAtorch::torch2Eigen::torchTensor2eigenMatrix<double>(U1);

    torch::Tensor rho = snap_rec.index({torch::indexing::Slice(), 2, torch::indexing::Slice()});
    std::cout << "rho: " << rho.sizes() << std::endl;
    Eigen::MatrixXd rhoEig = ITHACAtorch::torch2Eigen::torchTensor2eigenMatrix<double>(rho);

    torch::Tensor e = snap_rec.index({torch::indexing::Slice(), 3, torch::indexing::Slice()});
    std::cout << "e: " << e.sizes() << std::endl;
    Eigen::MatrixXd eEig = ITHACAtorch::torch2Eigen::torchTensor2eigenMatrix<double>(e);

    // Decompress
    std::cout << std::endl << " Decompress ..." << std::endl;
    torch::Tensor U0_dec = torch::matmul(U0, modesU0).unsqueeze(2);
    std::cout << "U0_dec: " << U0_dec.sizes() << std::endl;
    torch::Tensor U1_dec = torch::matmul(U1, modesU1).unsqueeze(2);
    std::cout << "U1_dec: " << U1_dec.sizes() << std::endl;
    torch::Tensor rho_dec = torch::matmul(rho, modesrho);
    std::cout << "rho_dec: " << rho_dec.sizes() << std::endl;
    torch::Tensor e_dec = torch::matmul(e, modese);
    std::cout << "e_dec: " << e_dec.sizes() << std::endl;

    // Save decompressed fields
    std::cout << std::endl << "Export decompressed fields ..." << std::endl;
    torch::Tensor U_dec = torch::cat({U0_dec, U1_dec, torch::zeros({U0_dec.sizes()[0], U0_dec.sizes()[1], 1})}, 2).view({U0_dec.sizes()[0], -1}).contiguous();
    auto U_list = ITHACAtorch::torch2Foam::torch2PtrList<vector>(U_dec);
    auto rho_list = ITHACAtorch::torch2Foam::torch2PtrList<scalar>(rho_dec);
    auto e_list = ITHACAtorch::torch2Foam::torch2PtrList<scalar>(e_dec);

    train_FOM.offlineSolveMach("./ITHACAoutput/Offline/Test/");

    // Convert to geometricfield rho
    auto field_rho =  train_FOM.rhofield[0];
    PtrList<volScalarField> rho_list_;

    for(int i=1; i<rho_list.size(); i++)
    {
        field_rho.ref().field() = rho_list[0];
        rho_list_.append(field_rho);
    }

    // Convert to geometricfield e
    auto field_e =  train_FOM.efield[0];
    PtrList<volScalarField> e_list_;

    for(int i=0; i<e_list.size(); i++)
    {
        field_e.ref().field() = e_list[i];
        e_list_.append(field_e);
    }

    // Convert to geometricfield U
    auto field_U =  train_FOM.Ufield[0];
    PtrList<volVectorField> U_list_;

    for(int i=0; i<U_list.size(); i++)
    {
        field_U.ref().field() = U_list[i];
        U_list_.append(field_U);
    }

    ITHACAstream::exportFields(U_list_, "U_decompressed", "U");
    ITHACAstream::exportFields(rho_list_, "rho_decompressed", "rho");
    ITHACAstream::exportFields(e_list_, "e_decompressed", "e");

    // Compression error U
    Info << endl << "Evaluate fields projection error ... "  << train_FOM.Ufield.size() << " " << U_list_.size() << endl;
    Eigen::MatrixXd errL2U = ITHACAutilities::errorL2Rel(train_FOM.Ufield, U_list_);
    // ITHACAstream::exportMatrix(errL2U, "errL2U", "matlab","./ITHACAoutput/ErrorsL2/");
    // cnpy::save(errL2U, "./ITHACAoutput/ErrorsL2/errL2U.npy");
    std::cout << "Mean U L2 compression error: " << errL2U.mean() << std::endl;
    // Compression error rho
    Info << endl << "Evaluate fields projection error ... "  << train_FOM.rhofield.size() << " " << rho_list_.size() << endl;
    Eigen::MatrixXd errL2rho = ITHACAutilities::errorL2Rel(train_FOM.rhofield, rho_list_);
    // ITHACAstream::exportMatrix(errL2rho, "errL2rho", "matlab","./ITHACAoutput/ErrorsL2/");
    // cnpy::save(errL2rho, "./ITHACAoutput/ErrorsL2/errL2rho.npy");
    std::cout << "Mean rho L2 compression error: " << errL2rho.mean() << std::endl;

    // Compression error e
    Info << endl << "Evaluate fields projection error ... "  << train_FOM.efield.size() << " " << e_list_.size() << endl;
    Eigen::MatrixXd errL2e = ITHACAutilities::errorL2Rel(train_FOM.efield, e_list_);
    // ITHACAstream::exportMatrix(errL2e, "errL2e", "matlab","./ITHACAoutput/ErrorsL2/");
    // cnpy::save(errL2e, "./ITHACAoutput/ErrorsL2/errL2e.npy");
    std::cout << "Mean e L2 compression error: " << errL2e.mean() << std::endl;


}

// void test_one_parameter_initial_velocity(tutorial00 test_FOM)
// {
//     // Read parameters from ITHACAdict file
//     ITHACAparameters *para = ITHACAparameters::getInstance(test_FOM._mesh(),
//                                                            test_FOM._runTime());
//     int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
//     int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
//     int NmodesUtest = para->ITHACAdict->lookupOrDefault<int>("NmodesUtest", 1);

//     /// Set the number of parameters
//     test_FOM.Pnumber = 1;
//     /// Set the dimension of the test set
//     test_FOM.Tnumber = NmodesUtest;

//     // Generate a number of Tnumber linearly equispaced samples inside the parameter range
//     Eigen::MatrixXd mu;
//     test_FOM.mu = cnpy::load(mu, "parTest.npy");

//     // Time parameters
//     test_FOM.startTime = 0;
//     test_FOM.finalTime = 2;
//     test_FOM.timeStep = 0.001;
//     test_FOM.writeEvery = 1;

//     // Perform The Offline Solve;
//     if (!ITHACAutilities::check_folder("./ITHACAoutput/Offline/Test/"))
//     {
//         test_FOM.offline = false;
//         // Info << "Offline Test data already exist, reading existing data" << endl;
//     }

//     test_FOM.offlineSolveInitialVelocity("./ITHACAoutput/Offline/Test/");
//     Eigen::MatrixXd trueSnapMatrix = Foam2Eigen::PtrList2Eigen(test_FOM.Ufield);

//     // Info << "snapshots size: " << trueSnapMatrix.size() << test_FOM.Ufield.size() << endl;
//     cnpy::save(trueSnapMatrix, "npTrueSnapshots.npy");

//     test_FOM.NUmodes = NmodesUproj;
//     ITHACAstream::read_fields(test_FOM.L_Umodes, "U", "./ITHACAoutput/POD_and_initial/", 1);
//     ITHACAstream::exportFields(test_FOM.L_Umodes, "./TEST", "uTest");

//     test_FOM.NL_Umodes = test_FOM.L_Umodes.size();
//     test_FOM.evaluateMatrices();

//     ReducedBurgers reduced_nonIntrusive(test_FOM);

//     // Set values of the reduced_nonIntrusive model
//     reduced_nonIntrusive.nu = 0.0001;
//     reduced_nonIntrusive.tstart = 0;
//     reduced_nonIntrusive.finalTime = 2;
//     reduced_nonIntrusive.dt = 0.001;
//     reduced_nonIntrusive.storeEvery = 0.001;
//     reduced_nonIntrusive.exportEvery = 0.001;
//     //reduced_nonIntrusive.Nphi_u = NmodesUproj;// the initial condition is added to the modes

//     Eigen::MatrixXd nonIntrusiveCoeff;

//     nonIntrusiveCoeff = cnpy::load(nonIntrusiveCoeff, "nonIntrusiveCoeff.npy", "rowMajor");

//     // Reconstruct the solution and export it
//     reduced_nonIntrusive.reconstruct(true, "./ITHACAoutput/ReconstructionNonIntrusive/", nonIntrusiveCoeff);

//     Eigen::MatrixXd errL2UnonIntrusive = ITHACAutilities::errorL2Rel(test_FOM.Ufield, reduced_nonIntrusive.uRecFields);

//     ITHACAstream::exportMatrix(errL2UnonIntrusive, "errL2UnonIntrusive", "matlab",
//                                "./ITHACAoutput/ErrorsL2/");
//     cnpy::save(errL2UnonIntrusive, "./ITHACAoutput/ErrorsL2/errL2UnonIntrusive.npy");

//     ReducedBurgers reduced_intrusive(test_FOM);

//     // Set values of the reduced model
//     reduced_intrusive.nu = 0.0001;
//     reduced_intrusive.tstart = 0;
//     reduced_intrusive.finalTime = 2;
//     reduced_intrusive.dt = 0.001;
//     reduced_intrusive.storeEvery = 0.001;
//     reduced_intrusive.exportEvery = 0.001;

//     reduced_intrusive.solveOnline(test_FOM.mu, 1);
//     ITHACAstream::exportMatrix(reduced_intrusive.online_solution, "red_coeff", "python", "./ITHACAoutput/red_coeff_intrusive");

//     // Reconstruct the solution and export it
//     reduced_intrusive.reconstruct(true, "./ITHACAoutput/ReconstructionIntrusive/");
//     ITHACAstream::exportFields(reduced_intrusive.uRecFields, "./ITHACAoutput/ReconstructionIntrusive/", "uRec");

//     Eigen::MatrixXd errL2Uintrusive = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
//                                                                   reduced_intrusive.uRecFields);

//     ITHACAstream::exportMatrix(errL2Uintrusive, "errL2UIntrusive", "matlab",
//                                "./ITHACAoutput/ErrorsL2/");
//     cnpy::save(errL2Uintrusive, "./ITHACAoutput/ErrorsL2/errL2UIntrusive.npy");

//     // Evaluate the true projection error
//     reduced_intrusive.trueProjection("./ITHACAoutput/Reconstruction/");

//     Eigen::MatrixXd errL2UtrueProjection = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
//                                                                        reduced_intrusive.uRecFields);

//     ITHACAstream::exportMatrix(errL2UtrueProjection, "errL2UtrueProjectionROM", "matlab",
//                                "./ITHACAoutput/ErrorsL2/");
//     cnpy::save(errL2UtrueProjection, "./ITHACAoutput/ErrorsL2/errL2UtrueProjectionROM.npy");
// }

// void nonlinear_one_parameter_initial_velocity(tutorial00 test_FOM)
// {
//     // Read parameters from ITHACAdict file
//     ITHACAparameters *para = ITHACAparameters::getInstance(test_FOM._mesh(),
//                                                            test_FOM._runTime());
//     int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
//     int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
//     int NmodesUtest = para->ITHACAdict->lookupOrDefault<int>("NmodesUtest", 100);
//     int NnonlinearModes = 2; //para->ITHACAdict->lookupOrDefault<int>("NnonlinearModes", 4);

//     /// Set the number of parameters
//     test_FOM.Pnumber = 1;

//     /// Set the dimension of the test set
//     test_FOM.Tnumber = NmodesUtest;

//     // load the test samples
//     Eigen::MatrixXd mu;
//     test_FOM.mu = cnpy::load(mu, "parTest.npy");

//     // Time parameters
//     test_FOM.startTime = 0;
//     test_FOM.finalTime = 2;
//     test_FOM.timeStep = 0.001;
//     test_FOM.writeEvery = 1;

//     Eigen::MatrixXd initial_latent;
//     cnpy::load(initial_latent, "./Autoencoders/ConvolutionalAe/latent_initial_2.npy");

//     std::cout << "LATENT INIT " << initial_latent << std::endl;

//     // Perform The Offline Solve;
//     if (!ITHACAutilities::check_folder("./ITHACAoutput/Offline/Test/"))
//     {
//         test_FOM.offline = false;
//         // Info << "Offline Test data already exist, reading existing data" << endl;
//     }

//     test_FOM.offlineSolveInitialVelocity("./ITHACAoutput/Offline/Test/");

//     // load modes from training
//     test_FOM.NUmodes = NmodesUproj;
//     ITHACAstream::read_fields(test_FOM.L_Umodes, "U", "./ITHACAoutput/POD_and_initial/", 1, NmodesUout);
//     test_FOM.NL_Umodes = test_FOM.L_Umodes.size();

//     // NM-LSPG
//     NonlinearReducedBurgers reduced_nm_lspg(test_FOM, "./Autoencoders/ConvolutionalAe/decoder_gpu_2.pt", NnonlinearModes, initial_latent);

//     // Set values of the reduced model
//     reduced_nm_lspg.nu = 0.0001;
//     reduced_nm_lspg.tstart = 0;
//     reduced_nm_lspg.finalTime = 2;
//     reduced_nm_lspg.dt = 0.001;
//     reduced_nm_lspg.storeEvery = 0.001;
//     reduced_nm_lspg.exportEvery = 0.001;

//     reduced_nm_lspg.solveOnline(test_FOM.mu, 1);
//     ITHACAstream::exportMatrix(reduced_nm_lspg.online_solution, "red_coeff", "python", "./ITHACAoutput/red_coeff_NM_LSPG");

//     // Reconstruct the solution and export it
//     ITHACAstream::exportFields(reduced_nm_lspg.uRecFields, "./ITHACAoutput/ReconstructionNMLSPG/", "uRec");
//     // reduced_nm_lspg.reconstruct(true,"./ITHACAoutput/ReconstructionNMLSPG/");

//     // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 482 #################### " << test_FOM.Ufield.size() << " " << reduced_nm_lspg.uRecFields.size() << " " << reduced_nm_lspg.online_solution.size() << endl;

//     Eigen::MatrixXd errL2UNMLSPG = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
//                                                                reduced_nm_lspg.uRecFields);

//     ITHACAstream::exportMatrix(errL2UNMLSPG, "errL2UNMLSPG", "matlab",
//                                "./ITHACAoutput/ErrorsL2/");
//     cnpy::save(errL2UNMLSPG, "./ITHACAoutput/ErrorsL2/errL2UNMLSPG.npy");
// }

// void nonlinear_test_rec(tutorial00 test_FOM)
// {
//     // Read parameters from ITHACAdict file
//     ITHACAparameters *para = ITHACAparameters::getInstance(test_FOM._mesh(),
//                                                            test_FOM._runTime());
//     int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
//     int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
//     int NmodesUtest = para->ITHACAdict->lookupOrDefault<int>("NmodesUtest", 100);
//     int NnonlinearModes = para->ITHACAdict->lookupOrDefault<int>("NnonlinearModes", 4);

//     /// Set the number of parameters
//     test_FOM.Pnumber = 1;

//     /// Set the dimension of the test set
//     test_FOM.Tnumber = NmodesUtest;

//     // load the test samples
//     Eigen::MatrixXd mu;
//     test_FOM.mu = cnpy::load(mu, "parTest.npy");

//     // Time parameters
//     test_FOM.startTime = 0;
//     test_FOM.finalTime = 2;
//     test_FOM.timeStep = 0.001;
//     test_FOM.writeEvery = 1;

//     Eigen::MatrixXd initial_latent;
//     cnpy::load(initial_latent, "./Autoencoders/ConvolutionalAe/latent_initial_4.npy");

//     // Perform The Offline Solve;
//     if (!ITHACAutilities::check_folder("./ITHACAoutput/Offline/Test/"))
//     {
//         test_FOM.offline = false;
//         // Info << "Offline Test data already exist, reading existing data" << endl;
//     }

//     test_FOM.offlineSolveInitialVelocity("./ITHACAoutput/Offline/Test/");

//     // load modes from training
//     test_FOM.NUmodes = NmodesUproj;
//     ITHACAstream::read_fields(test_FOM.L_Umodes, "U", "./ITHACAoutput/POD_and_initial/", 1, NmodesUout);
//     test_FOM.NL_Umodes = test_FOM.L_Umodes.size();

//     // load the test snapshots
//     // Generate your data set. At this point you can add transforms to you data
//     // set, e.g. stack your batches into a single tensor.
//     Eigen::MatrixXd dataset_eig;
//     torch::Tensor inputs_true = torch2Eigen::eigenMatrix2torchTensor(cnpy::load(dataset_eig, "./npTrueSnapshots_framed.npy"));

//     // load autoencoder
//     auto ae = autoPtr<torch::jit::script::Module>(new torch::jit::script::Module(torch::jit::load("./Autoencoders/ConvolutionalAe/model_gpu_4.pt")));

//     // forward autoencoder
//     // auto rec_torch = torch::zeros({2001, 7200});
//     std::vector<torch::jit::IValue> input;
//     input.push_back(inputs_true.reshape({2001, 2, 60, 60}).to(torch::kCUDA));
//     torch::Tensor forward_tensor = ae->forward(input).toTensor().squeeze().detach().to(torch::kCPU);

//     auto tensor_stacked = torch::cat({forward_tensor.reshape({2001, 2, 60, 60}), torch::zeros({2001, 1, 60, 60})}, 1).reshape({2001, 3, -1}).transpose(1, 2).contiguous();

//     PtrList<volVectorField> rec_fields;
//     for (int i = 0; i < 2001; i++)
//     {
//         auto g = autoPtr<volVectorField>(new volVectorField(test_FOM._U0()));
//         auto single_snap = tensor_stacked.slice(0, i, 1 + i).squeeze();
//         auto field = torch2Foam::torch2Field<vector>(single_snap);
//         g.ref().ref().field() = field;
//         rec_fields.append(g);
//     }
//     ITHACAstream::exportFields(rec_fields, "./REC_AE", "g0");

//     // compute LRelError
//     Eigen::MatrixXd errConsistency = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
//                                                                  rec_fields);

//     ITHACAstream::exportMatrix(errConsistency, "errConsistency", "matlab",
//                                "./ITHACAoutput/ErrorsL2/");
//     cnpy::save(errConsistency, "./ITHACAoutput/ErrorsL2/errConsistency.npy");
// }

// void nonlinear_test_data(tutorial00 test_FOM)
// {
//     // Read parameters from ITHACAdict file
//     ITHACAparameters *para = ITHACAparameters::getInstance(test_FOM._mesh(),
//                                                            test_FOM._runTime());
//     int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
//     int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
//     int NmodesUtest = para->ITHACAdict->lookupOrDefault<int>("NmodesUtest", 100);
//     int NnonlinearModes = para->ITHACAdict->lookupOrDefault<int>("NnonlinearModes", 4);

//     /// Set the number of parameters
//     test_FOM.Pnumber = 1;

//     /// Set the dimension of the test set
//     test_FOM.Tnumber = NmodesUtest;

//     // load the test samples
//     Eigen::MatrixXd mu;
//     test_FOM.mu = cnpy::load(mu, "parTest.npy");

//     // Time parameters
//     test_FOM.startTime = 0;
//     test_FOM.finalTime = 2;
//     test_FOM.timeStep = 0.001;
//     test_FOM.writeEvery = 1;

//     Eigen::MatrixXd initial_latent;
//     initial_latent = cnpy::load(initial_latent, "./Autoencoders/ConvolutionalAe/latent_initial_4.npy", "rowMajor");
//     std::cout << "INITIAL" << initial_latent << std::endl;

//     // Perform The Offline Solve;
//     if (!ITHACAutilities::check_folder("./ITHACAoutput/Offline/Test/"))
//     {
//         test_FOM.offline = false;
//         // Info << "Offline Test data already exist, reading existing data" << endl;
//     }

//     test_FOM.offlineSolveInitialVelocity("./ITHACAoutput/Offline/Test/");

//     // load modes from training
//     test_FOM.NUmodes = NmodesUproj;
//     ITHACAstream::read_fields(test_FOM.L_Umodes, "U", "./ITHACAoutput/POD_and_initial/", 1, NmodesUout);
//     test_FOM.NL_Umodes = test_FOM.L_Umodes.size();

//     // load the test snapshots
//     Eigen::MatrixXd dataset_eig;
//     dataset_eig = cnpy::load(dataset_eig, "./nonIntrusiveCoeffConvAe.npy", "rowMajor");
//     std::cout << "EIG" << dataset_eig << std::endl;
//     torch::Tensor inputs_hidden = torch2Eigen::eigenMatrix2torchTensor(dataset_eig);
//     std::cout << "TEST" << inputs_hidden.to(at::kFloat) << std::endl;

//     // load autoencoder
//     auto decoder = autoPtr<torch::jit::script::Module>(new torch::jit::script::Module(torch::jit::load("./Autoencoders/ConvolutionalAe/decoder_gpu_4.pt")));

//     // define initial velocity field and initialize decoder output variable g0
//     auto _g0 = autoPtr<volVectorField>(new volVectorField(test_FOM._U0()));

//     // declare input of decoder of type IValue since the decoder is loaded from pytorch
//     std::vector<torch::jit::IValue> input_init;
//     torch::Tensor latent_initial_tensor = torch2Eigen::eigenMatrix2torchTensor(initial_latent);
//     std::cout << "TEST INIT" << latent_initial_tensor << std::endl;

//     input_init.push_back(latent_initial_tensor.to(at::kFloat).to(torch::kCUDA));

//     torch::Tensor tensor = decoder->forward(std::move(input_init)).toTensor().squeeze().detach().to(torch::kCPU);

//     auto tensor_stacked_init = torch::cat({std::move(tensor).reshape({2, 60, 60}), torch::zeros({1, 60, 60})}, 0).reshape({3, -1}).transpose(0, 1).contiguous();
//     auto g0 = torch2Foam::torch2Field<vector>(tensor_stacked_init);
//     _g0.ref().ref().field() = std::move(g0);
//     PtrList<volVectorField> save_field;
//     save_field.append(_g0());
//     ITHACAstream::exportFields(save_field, "./REFtest", "g0");

//     // forward autoencoder
//     std::vector<torch::jit::IValue> input;
//     input.push_back(inputs_hidden.to(at::kFloat).to(torch::kCUDA));

//     torch::Tensor forward_tensor = decoder->forward(input).toTensor().squeeze().detach().to(torch::kCPU);

//     auto tensor_stacked = torch::cat({forward_tensor.reshape({2001, 2, 60, 60}), torch::zeros({2001, 1, 60, 60})}, 1).reshape({2001, 3, -1}).transpose(1, 2).contiguous();

//     PtrList<volVectorField> rec_fields;
//     for (int i = 0; i < 2001; i++)
//     {
//         auto g = autoPtr<volVectorField>(new volVectorField(test_FOM._U0()));
//         auto single_snap = tensor_stacked.slice(0, i, 1 + i).squeeze();
//         auto field = torch2Foam::torch2Field<vector>(single_snap);
//         g.ref().ref().field() = field;
//         rec_fields.append(g);
//     }
//     ITHACAstream::exportFields(rec_fields, "./INTR", "g0");

//     // compute LRelError
//     Eigen::MatrixXd errConsistency = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
//                                                                  rec_fields);

//     ITHACAstream::exportMatrix(errConsistency, "errConsistency", "matlab",
//                                "./ITHACAoutput/ErrorsL2/");
//     cnpy::save(errConsistency, "./ITHACAoutput/ErrorsL2/errConsistency.npy");
// }

// void nonlinear_one_parameter_initial_velocity_hr(tutorial00 test_FOM)
// {
//     // Read parameters from ITHACAdict file
//     ITHACAparameters *para = ITHACAparameters::getInstance(test_FOM._mesh(),
//                                                            test_FOM._runTime());
//     int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
//     int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
//     int NmodesUtest = para->ITHACAdict->lookupOrDefault<int>("NmodesUtest", 100);
//     int NDEIM = para->ITHACAdict->lookupOrDefault<int>("NDEIM", 15);
//     int NnonlinearModes = 4; //para->ITHACAdict->lookupOrDefault<int>("NnonlinearModes", 4);

//     /// Set the number of parameters
//     test_FOM.Pnumber = 1;

//     /// Set the dimension of the test set
//     test_FOM.Tnumber = NmodesUtest;

//     // load the test samples
//     Eigen::MatrixXd mu;
//     test_FOM.mu = cnpy::load(mu, "parTest.npy");

//     // Time parameters
//     test_FOM.startTime = 0;
//     test_FOM.finalTime = 2;
//     test_FOM.timeStep = 0.001;
//     test_FOM.writeEvery = 1;

//     Eigen::MatrixXd initial_latent;
//     cnpy::load(initial_latent, "./Autoencoders/ConvolutionalAe/latent_initial_4.npy");

//     std::cout << "LATENT INIT " << initial_latent << std::endl;

//     // // Perform The Offline Solve;
//     // if (!ITHACAutilities::check_folder("./ITHACAoutput/Offline/Test/"))
//     // {
//     //     test_FOM.offline = false;
//         Info << "Offline Test data already exist, reading existing data" << endl;
//     // }

//     test_FOM.offlineSolveInitialVelocity("ITHACAoutput/Offline/Test/");

//     // load modes from training
//     test_FOM.NUmodes = NmodesUproj;
//     ITHACAstream::read_fields(test_FOM.L_Umodes, "U", "./ITHACAoutput/POD_and_initial/", 1, NmodesUout);
//     test_FOM.NL_Umodes = test_FOM.L_Umodes.size();

//     // NM-LSPG
//     NonlinearReducedBurgers reduced_nm_lspg(test_FOM, "./Autoencoders/ConvolutionalAe/decoder_gpu_4.pt", NnonlinearModes, initial_latent);

//     // Set values of the reduced model
//     reduced_nm_lspg.nu = 0.0001;
//     reduced_nm_lspg.tstart = 0;
//     reduced_nm_lspg.finalTime = 2;
//     reduced_nm_lspg.dt = 0.001;
//     reduced_nm_lspg.storeEvery = 0.01;       // do not store and export
//     reduced_nm_lspg.exportEvery = 10;      // do not store and export
//     reduced_nm_lspg.exportResidual = 0.01; // save the residuals for hr

//     reduced_nm_lspg.solveOnline(test_FOM.mu, 1);
//     ITHACAstream::exportFields(reduced_nm_lspg.residualsList, "./RESIDUAL", "res");

//     // Create DEIM object with given number of basis functions
//     DEIM<volScalarField> c(reduced_nm_lspg.residualsList, NDEIM, "residuals");

//     // Generate the submeshes with the depth of the layer
//     auto fields = c.generateSubmeshes(1, test_FOM._mesh(), test_FOM.L_Umodes[0]);

//     // // Define a new online parameter
//     // Eigen::MatrixXd par_new(2, 1);
//     // par_new(0, 0) = 0;
//     // par_new(1, 0) = 0;

//     // // Online evaluation of the non linear function
//     // Eigen::VectorXd aprfield = c.MatrixOnline * c.onlineCoeffs(par_new);

//     // // Transform to an OpenFOAM field and export
//     // volScalarField S2("S_online", Foam2Eigen::Eigen2field(S, aprfield));
//     // ITHACAstream::exportSolution(S2, name(1), "./ITHACAoutput/Online/");

//     // // Evaluate the full order function and export it
//     // DEIM_function::evaluate_expression(S, par_new);
//     // ITHACAstream::exportSolution(S, name(1), "./ITHACAoutput/Online/");

//     // // Compute the L2 error and print it
//     Info << ITHACAutilities::errorL2Rel(S2, S) << endl;

//     // ITHACAstream::exportMatrix(reduced_nm_lspg.online_solution, "red_coeff", "python", "./ITHACAoutput/red_coeff_NM_LSPG");

//     // // Reconstruct the solution and export it
//     // ITHACAstream::exportFields(reduced_nm_lspg.uRecFields, "./ITHACAoutput/ReconstructionNMLSPG/", "uRec");
//     // // reduced_nm_lspg.reconstruct(true,"./ITHACAoutput/ReconstructionNMLSPG/");

//     Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 482 #################### " << test_FOM.Ufield.size() << " " <<  reduced_nm_lspg.uRecFields.size() << " " << reduced_nm_lspg.online_solution.size() << endl;

//     // Eigen::MatrixXd errL2UNMLSPG = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
//     //                          reduced_nm_lspg.uRecFields);

//     // ITHACAstream::exportMatrix(errL2UNMLSPG, "errL2UNMLSPG", "matlab",
//     //                            "./ITHACAoutput/ErrorsL2/");
//     // cnpy::save(errL2UNMLSPG, "./ITHACAoutput/ErrorsL2/errL2UNMLSPG.npy");
// }