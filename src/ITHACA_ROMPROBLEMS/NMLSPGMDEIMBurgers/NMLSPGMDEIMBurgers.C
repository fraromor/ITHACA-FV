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

#include "NMLSPGMDEIMBurgers.H"

using namespace ITHACAtorch;

// * * * * * * * * * * * * * * * Embeddings * * * * * * * * * * * * * * * * //
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
    initial_mdeim = torch2Eigen::torchTensor2eigenMatrix<double>(forwarded);
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

    Eigen::VectorXd push_forward = torch2Eigen::torchTensor2eigenMatrix<double>(push_forward_tensor);

    // add reference term
    return push_forward;
}

// * * * * * * * * * * * * * * * newton_burgers * * * * * * * * * * * * * * * * //
// private method used only inside EmbeddingMDEIM::forward. Return reference element of embedding s.t. initial embedding is mu * _U0()
Eigen::VectorXd newton_nmlspg_mdeim_burgers::embedding_ref(const scalar mu) const
{
    return mu * U0_mdeim - embedding->initial_mdeim;
}

// Operator to evaluate the residual
int newton_nmlspg_mdeim_burgers::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
{
    std::cout << " residual, x = " << x.transpose() << std::endl;

    auto g = embedding->forward(x, mu) + embedding_ref(mu);

    // TODO avoid to resource to full field to evaluate resEqn
    volVectorField &a_tmp = createFieldFromMagicPoints(g);
    fvMesh &mesh = problem->_mesh();

    volVectorField &tmp = a_tmp.oldTime();
    tmp = createFieldFromMagicPoints(g_old);

    auto phi = linearInterpolate(a_tmp) & mesh.Sf();

    fvVectorMatrix resEqn(
        fvm::ddt(a_tmp) + 0.5 * fvm::div(phi, a_tmp) - fvm::laplacian(dimensionedScalar(dimViscosity, nu.value()), a_tmp));

    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Foam2Eigen::fvMatrix2Eigen(resEqn, A, b);

    Eigen::MatrixXd A_mdeim = projectMatrix(A);// shape is n_mdeim x n_mdeim
    Eigen::VectorXd b_deim = projectVector(b);//shape is n_deim

    //TODO multiply by pinv somehow
    Eigen::VectorXd g_ = A_mdeim * g;
    fvec =  g_ - pinvVector * b_deim;
    return 0;
}

// Offline
void newton_nmlspg_mdeim_burgers::initPseudoInverses(std::tuple<List<Eigen::SparseMatrix<double>>, List<Eigen::VectorXd>> matrixModes)
{
    pinvVector.resize(std::get<1>(matrixModes)[0].size(), std::get<1>(matrixModes).size());

    for (int j = 0; j < std::get<1>(matrixModes).size(); j++)
    {
        for (int i = 0; i < mdeim->fieldsB.size(); i++)
        {
            int ind_row = mdeim->localMagicPointsB[i] + mdeim->xyz_B[i] * mdeim->fieldsB[i].size();
            pinvVector.coeffRef(i, j) = std::get<1>(matrixModes)[j](ind_row);
        }
    }

    pinvMatrix.resize(std::get<0>(matrixModes)[0].rows()*std::get<0>(matrixModes)[0].cols(), std::get<0>(matrixModes).size());

    for (int j = 0; j < std::get<0>(matrixModes).size(); j++)
    {
        for (int i = 0; i < mdeim->fieldsA.size(); i++)
        {
            int ind_row = mdeim->localMagicPointsA[i].first() + mdeim->xyz_A[i].first() *
                            mdeim->fieldsA[i].size();
            int ind_col = mdeim->localMagicPointsA[i].second() + mdeim->xyz_A[i].second() *
                            mdeim->fieldsA[i].size();
            pinvMatrix.coeffRef(i, j) = std::get<0>(matrixModes)[j].coeffRef(ind_row, ind_col);
        }
    }

}
// Online
Eigen::MatrixXd newton_nmlspg_mdeim_burgers::projectMatrix(Eigen::MatrixXd full_matrix) const
{
    Eigen::MatrixXd hr_matrix(mdeim->fieldsA.size(), 1);

    for (int i = 0; i < mdeim->fieldsA.size(); i++)
    {
        int ind_row = mdeim->localMagicPointsA[i].first() + mdeim->xyz_A[i].first() *
                        mdeim->fieldsA[i].size();
        int ind_col = mdeim->localMagicPointsA[i].second() + mdeim->xyz_A[i].second() *
                        mdeim->fieldsA[i].size();
        hr_matrix(i) = full_matrix.coeffRef(ind_row, ind_col);
    }
    return hr_matrix;
}
// Online
Eigen::VectorXd newton_nmlspg_mdeim_burgers::projectVector(Eigen::VectorXd full_vector) const
{
    Eigen::MatrixXd hr_vector(mdeim->fieldsB.size(), 1);

    for (int i = 0; i < mdeim->fieldsB.size(); i++)
    {
        int ind_row = mdeim->localMagicPointsB[i] + mdeim->xyz_B[i] * mdeim->fieldsB[i].size();
        hr_vector(i) = full_vector(ind_row);
    }
    return hr_vector;
}
// Online
volVectorField& newton_nmlspg_mdeim_burgers::createFieldFromMagicPoints(Eigen::VectorXd deim_field) const
{
    Eigen::VectorXd ret_field = Foam2Eigen::field2Eigen(embedding->_U0());
    for (int i = 0; i < mdeim->fieldsB.size(); i++)
    {
        int ind_row = mdeim->localMagicPointsB[i] + mdeim->xyz_B[i] * mdeim->fieldsB[i].size();
        ret_field(ind_row) = deim_field(i); //TODO check
    }

    volVectorField& foam_field = embedding->_U0();
    foam_field = Foam2Eigen::Eigen2field(foam_field, ret_field);
    return foam_field;
}

// * * * * * * * * * * * * * * * NMLSPGMDEIMBurgers * * * * * * * * * * * * * * * * //
// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

NMLSPGMDEIMBurgers::NMLSPGMDEIMBurgers()
{
    para = ITHACAparameters::getInstance();
}

NMLSPGMDEIMBurgers::NMLSPGMDEIMBurgers(Burgers &FOMproblem, fileName decoder_path, int dim_latent, Eigen::MatrixXd latent_initial)
    : Nphi_u{dim_latent},
      problem{&FOMproblem}
{

    // problem->L_Umodes is used to create volVectorFields
    embedding_mdeim = autoPtr<EmbeddingMDEIM>(new EmbeddingMDEIM(Nphi_u, decoder_path, problem->L_Umodes[0], latent_initial));

    // ! newton_object_mdeim is initialized in NMLSPG_MMDEIM because the
    // dimensions of MDEIM have to be passed
}

void NMLSPGMDEIMBurgers::NMLSPG_MMDEIM(int NmodesU, int NmodesDEIMA, int NmodesDEIMB)
{
    // FOMproblem is only needed for initial conditions
    NUmodes = NmodesU;
    NmodesDEIMA = NmodesDEIMA;
    NmodesDEIMB = NmodesDEIMB;

    newton_object_mdeim = newton_nmlspg_mdeim_burgers(Nphi_u, NUmodes, *problem, embedding_mdeim.ref(), problem->L_Umodes[0]);
    newton_object_mdeim.mdeim = autoPtr<DEIM_burgers>(new DEIM_burgers(matrixMDEIMList, NmodesDEIMA, NmodesDEIMB, "U_matrix"));
    fvMesh &mesh = const_cast<fvMesh &>(problem->L_Umodes[0].mesh());
    // Differential Operator
    newton_object_mdeim.mdeim->fieldsA = newton_object_mdeim.mdeim->generateSubmeshesMatrix(2, mesh, problem->L_Umodes[0]);
    // Source Terms
    newton_object_mdeim.mdeim->fieldsB = newton_object_mdeim.mdeim->generateSubmeshesVector(2, mesh, problem->L_Umodes[0]);

    newton_object_mdeim.initPseudoInverses(newton_object_mdeim.mdeim->Matrix_Modes);
    newton_object_mdeim.U0_mdeim = newton_object_mdeim.projectVector(Foam2Eigen::field2Eigen(problem->L_Umodes[0]));
}

void NMLSPGMDEIMBurgers::OnlineSolveMDEIM(Eigen::MatrixXd mu, int startSnap)
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
    M_Assert(ITHACAutilities::isInteger(exportMatricesMDEIM / dt) == true,
             "The variable storeEvery must be an integer multiple of the time step dt.");
    M_Assert(ITHACAutilities::isInteger(exportEvery / storeEvery) == true,
             "The variable exportEvery must be an integer multiple of the variable storeEvery.");
    // Info << " # DEBUG NMLSPGMDEIMBurgers_central.C, line 266 # " << endl;
    int numberOfStores = round(storeEvery / dt);

    // numberOfResiduals defaults to 0 and in that case no residual is saved
    int numberOfResiduals = round(exportResidual / dt);
    int numberOfMatricesMDEIM = round(exportMatricesMDEIM / dt);

    // Counter of the number of online solutions saved, accounting also time as parameter
    int counter2 = 0;

    // Set number of online solutions
    int Ntsteps = static_cast<int>((finalTime - tstart) / dt);
    int onlineSizeTimeSeries = static_cast<int>(Ntsteps / numberOfStores);

    // resize the online solution list with the length of n_parameters times
    // length of the time series
    online_solution.resize((mu.cols()) * (onlineSizeTimeSeries));
    // Info << " # DEBUG NMLSPGMDEIMBurgers_central.C, line 282 # " << endl;
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

        // Info << " # DEBUG NMLSPGMDEIMBurgers_central.C, line 296 # " << Nphi_u << endl;
        // Create and resize the solution vector (column vector)
        y.resize(Nphi_u, 1);
        y = embedding_mdeim->latent_initial.transpose();
        newton_object_mdeim.g_old = embedding_mdeim->forward(y, mu(0, n_param))+ newton_object_mdeim.embedding_ref(mu(0, n_param));
        // Info << " # DEBUG NMLSPGMDEIMBurgers_central.C, line 301 # " << endl;
        // TODO add flag to avoid saving the full reconstructed field
        volVectorField tmp = newton_object_mdeim.createFieldFromMagicPoints(newton_object_mdeim.g_old);
        uRecFields.append(tmp.clone());
        // Info << " # DEBUG NMLSPGMDEIMBurgers_central.C, line 305 # " << endl;
        // Set some properties of the newton object
        newton_object_mdeim.mu = mu(0, n_param);
        newton_object_mdeim.nu = nu;
        newton_object_mdeim.dt = dt;
        newton_object_mdeim.tauU = tauU;

        // Create vector to store temporal solution and save initial condition
        // as first solution
        // Info << " # DEBUG NMLSPGMDEIMBurgers_central.C, line 314 # " << endl;
        Eigen::MatrixXd tmp_sol(Nphi_u + 1, 1);
        tmp_sol(0) = time;
        // Info << " # DEBUG NMLSPGMDEIMBurgers_central.C, line 317 # " << endl;
        tmp_sol.col(0).tail(y.rows()) = y;
        // Info << " # DEBUG NMLSPGMDEIMBurgers_central.C, line 319 # " << endl;
        online_solution[counter2] = tmp_sol;
        counter2++;
        counter++;
        nextStore += numberOfStores;

        // Create nonlinear solver object
        // Info << " # DEBUG NMLSPGMDEIMBurgers_central.C, line 326 # " << endl;
        // Create nonlinear solver object
        Eigen::NumericalDiff<newton_nmlspg_mdeim_burgers, Eigen::Central> numDiffobject(newton_object_mdeim, 1.e-05);
        Eigen::LevenbergMarquardt<decltype(numDiffobject)> lm(numDiffobject);
        // Info << " # DEBUG NMLSPGMDEIMBurgers_central.C, line 330 # " << endl;
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
            // Info << " # DEBUG NMLSPGMDEIMBurgers_central.C, line 347 # " << endl;
            Eigen::LevenbergMarquardtSpace::Status ret = lm.minimize(y);

            std::cout << "LM finished with status: " << ret << std::endl;

            std::cout << " minimum: " << y.transpose() << endl;

            Eigen::VectorXd res(NUmodes);
            res.setZero();

            // update the old solution for the evaluation of the residual
            numDiffobject(y, res);
            numDiffobject.g_old = embedding_mdeim->forward(y, mu(0, n_param)) + numDiffobject.embedding_ref(mu(0, n_param));

            // TODO add flag to avoid saving the full reconstructed field
            volVectorField tmp = numDiffobject.createFieldFromMagicPoints(numDiffobject.g_old);
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

            counter++;
            time = time + dt;
        }
    }
}

// * * * * * * * * * * * * * * *  Evaluation  * * * * * * * * * * * * * //

void NMLSPGMDEIMBurgers::reconstruct(bool exportFields, fileName folder)
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
void NMLSPGMDEIMBurgers::reconstruct(bool exportFields, fileName folder, Eigen::MatrixXd redCoeff)
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
