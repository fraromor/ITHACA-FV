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

#include "NonlinearReducedBurgers_central.H"

using namespace ITHACAtorch;
// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

NonlinearReducedBurgers::NonlinearReducedBurgers()
{
    para = ITHACAparameters::getInstance();
}

NonlinearReducedBurgers::NonlinearReducedBurgers(Burgers &FOMproblem, fileName decoder_path, int dim_latent, Eigen::MatrixXd latent_initial)
    : Nphi_u{dim_latent},
      problem{&FOMproblem}
{

    // problem->L_Umodes is used to create volVectorFields
    embedding = autoPtr<Embedding>(new Embedding(Nphi_u, decoder_path, problem->L_Umodes[0], latent_initial));

    // FOMproblem is only needed for initial conditions
    newton_object = newton_nmlspg_burgers(Nphi_u, 2 * embedding->output_dim, FOMproblem, embedding.ref(), problem->L_Umodes[0]);
}

Embedding::Embedding(int dim, fileName decoder_path, volVectorField &U0, Eigen::MatrixXd lat_init) : latent_dim{dim}, latent_initial{lat_init}
{
    // get the number of degrees of freedom relative to a single component
    output_dim = U0.size(); // 3600
    Info << " # DEBUG NonlinearReducedBurgers_central.C, line 69 # " << endl;
    decoder = autoPtr<torch::jit::script::Module>(new torch::jit::script::Module(torch::jit::load(decoder_path)));
    Info << " # DEBUG NonlinearReducedBurgers_central.C, line 71 # " << endl;
    // define initial velocity field _U0 used to define the reference snapshot
    // and initialize decoder output variable g0
    _U0 = autoPtr<volVectorField>(new volVectorField(U0));
    _g0 = autoPtr<volVectorField>(new volVectorField(U0));

    // declare input of decoder of type IValue since the decoder is loaded from pytorch
    std::vector<torch::jit::IValue> input;
    torch::Tensor latent_initial_tensor = torch2Eigen::eigenMatrix2torchTensor(latent_initial);

    // the tensor inputs of the decoder must be of type at::kFloat (not double)
    input.push_back(latent_initial_tensor.to(at::kFloat).to(torch::kCUDA));

    std::cout << "LATENT INITIAL" << latent_initial_tensor << std::endl;

    torch::Tensor tensor = decoder->forward(std::move(input)).toTensor().to(torch::kCPU);

    // add the z component to the tensor as a zero {1, 60, 60} tensor and
    // reshape the tensor s.t. the components x,y,z of a single cell center are
    // contiguous in memory (this is necessary for torch2field method)
    auto tensor_stacked = torch::cat({std::move(tensor).reshape({2, 60, 60}), torch::zeros({1, 60, 60})}, 0).reshape({3, -1}).transpose(0, 1).contiguous();

    // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 94 # " << endl;
    auto g0 = torch2Foam::torch2Field<vector>(tensor_stacked);
    _g0.ref().ref().field() = std::move(g0);
    // save_field.append(_g0.clone();
    // ITHACAstream::exportFields(save_field, "./REF", "g0");
}

// private method used only inside Embedding::forward. Return reference element of embedding s.t. initial embedding is mu * _U0()
autoPtr<volVectorField> Embedding::embedding_ref(const scalar mu)
{
    return autoPtr<volVectorField>(new volVectorField(mu * _U0() - _g0()));
}

autoPtr<volVectorField> Embedding::forward(const Eigen::VectorXd &x, const scalar mu)
{
    // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 109 # " << endl;

    // declare input of decoder of type IValue since the decoder is loaded from pytorch
    std::vector<torch::jit::IValue> input;
    Eigen::MatrixXd input_matrix{x};

    input_matrix.resize(1, latent_dim);
    torch::Tensor input_tensor = torch2Eigen::eigenMatrix2torchTensor(std::move(input_matrix));
    input_tensor = input_tensor.reshape({1, latent_dim});
    input_tensor = input_tensor.set_requires_grad(true);
    // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 119 # " << endl;
    // the tensor inputs of the decoder must be of type at::kFloat (not double)
    input.push_back(input_tensor.to(at::kFloat).to(torch::kCUDA));

    torch::Tensor push_forward_tensor = decoder->forward(std::move(input)).toTensor().to(torch::kCPU);
    // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 124 # " << endl;
    auto g = autoPtr<volVectorField>(new volVectorField(_U0()));

    // add the z component to the tensor as a zero {1, 60, 60} tensor and
    // reshape the tensor s.t. the components x,y,z of a single cell center are
    // contiguous in memory (this is necessary for torch2field method)
    auto tensor_stacked = torch::cat({push_forward_tensor.reshape({2, 60, 60}), torch::zeros({1, 60, 60})}, 0).reshape({3, -1}).transpose(0, 1).contiguous();
    // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 131 # " << endl;
    auto push_forward = torch2Foam::torch2Field<vector>(tensor_stacked);

    // add reference term
    g.ref().ref().field() = std::move(push_forward);
    g.ref() += embedding_ref(mu).ref();
    // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 137 # " << endl;
    // save_field.append(g().clone());
    // if (counter == 10) {
    //     std::cout << "SAVED" << std::endl;
    //     ITHACAstream::exportFields(save_field, "./Forwarded","g");
    // }

    return g;
}

// Operator to evaluate the residual
int newton_nmlspg_burgers::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
{
    std::cout << " residual, x = " << x.transpose() << std::endl;

    auto g = embedding->forward(x, mu);
    volVectorField &a_tmp = g();
    fvMesh &mesh = problem->_mesh();

    auto a_old = g_old();
    volVectorField &tmp = a_tmp.oldTime();
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
    // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 266 # " << endl;
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
    // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 282 # " << endl;
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
        int counterMatrixMDEIM = 1;

        // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 296 # " << Nphi_u << endl;
        // Create and resize the solution vector (column vector)
        y.resize(Nphi_u, 1);
        y = embedding->latent_initial.transpose();
        newton_object.g_old = embedding->forward(y, mu(0, n_param));
        // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 301 # " << endl;
        auto tmp = newton_object.g_old();
        // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 303 # " << endl;
        uRecFields.append(tmp.clone());
        // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 305 # " << endl;
        // Set some properties of the newton object
        newton_object.mu = mu(0, n_param);
        newton_object.nu = nu;
        newton_object.dt = dt;
        newton_object.tauU = tauU;

        // Create vector to store temporal solution and save initial condition
        // as first solution
        // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 314 # " << endl;
        Eigen::MatrixXd tmp_sol(Nphi_u + 1, 1);
        tmp_sol(0) = time;
        // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 317 # " << endl;
        tmp_sol.col(0).tail(y.rows()) = y;
        // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 319 # " << endl;
        online_solution[counter2] = tmp_sol;
        counter2++;
        counter++;
        nextStore += numberOfStores;

        // Create nonlinear solver object
        // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 326 # " << endl;
        // Create nonlinear solver object
        Eigen::NumericalDiff<newton_nmlspg_burgers, Eigen::Central> numDiffobject(newton_object, 1.e-05);
        Eigen::LevenbergMarquardt<decltype(numDiffobject)> lm(numDiffobject);
        // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 330 # " << endl;
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
            // Info << " # DEBUG NonlinearReducedBurgers_central.C, line 347 # " << endl;
            Eigen::LevenbergMarquardtSpace::Status ret = lm.minimize(y);

            std::cout << "LM finished with status: " << ret << std::endl;

            std::cout << " minimum: " << y.transpose() << endl;

            Eigen::VectorXd res(2 * numDiffobject.embedding->output_dim);
            res.setZero();

            // update the old solution for the evaluation of the residual
            numDiffobject(y, res);
            numDiffobject.g_old = embedding->forward(y, mu(0, n_param));
            auto tmp = numDiffobject.g_old();
            uRecFields.append(tmp.clone());

            std::cout << "################## Online solve N° " << counter << " ##################" << std::endl;
            // Info << "Time = " << time << endl;

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
