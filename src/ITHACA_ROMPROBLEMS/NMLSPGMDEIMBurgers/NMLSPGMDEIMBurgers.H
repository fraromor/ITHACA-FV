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
Class
    reducedSteadyNS
Description
    A reduced problem for the stationary NS equations
SourceFiles
    reducedSteadyNS.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the reducedSteadyNS class.
/// \dir
/// Directory containing the header and source files for the reducedSteadyNS class

#ifndef NMLSPGMDEIMBurgers_H
#define NMLSPGMDEIMBurgers_H

#include <torch/script.h>
#include <torch/torch.h>
#include "torch2Eigen.H"
#include "torch2Foam.H"

#include "Foam2Eigen.H"
#include "fvCFD.H"
#include "IOmanip.H"
#include "ReducedProblem.H"
#include "NonlinearReducedBurgers_central.H"
#include "burgers.H"
#include "DEIM.H"
#include "ITHACAutilities.H"
#include "EigenFunctions.H"
#include <Eigen/Eigen>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <torch/script.h>
#include <utility>

class EmbeddingMDEIM
{
public:
    /// latent dimension
    int latent_dim;

    /// decoder module
    autoPtr<torch::jit::script::Module> decoder;

    /// Initial condition
    autoPtr<volVectorField> _U0;

    /// initial latent variable
    Eigen::MatrixXd latent_initial;

    /// mdeim restricted latent variables image
    Eigen::MatrixXd initial_mdeim;

    /// mdeim restricted initial field
    Eigen::VectorXd U0_mdeim;

    /// DEBUG commodity counter
    int counter = 0;
    PtrList<volVectorField> save_field;

    EmbeddingMDEIM() {}
    EmbeddingMDEIM(int latent_dim, fileName decoder_path, volVectorField &U0, Eigen::MatrixXd lat_init);

    EmbeddingMDEIM(EmbeddingMDEIM &embedding) : latent_dim{embedding.latent_dim},
                                                decoder{embedding.decoder},
                                                _U0{embedding._U0} {};

    EmbeddingMDEIM(EmbeddingMDEIM &&embedding) : latent_dim{std::move(embedding.latent_dim)},
                                                 decoder{std::move(embedding.decoder)},
                                                 _U0{std::move(embedding._U0)} {};

    Eigen::VectorXd forward(const Eigen::VectorXd &x, const scalar mu);
    Eigen::VectorXd embedding_ref(const scalar mu);

    ~EmbeddingMDEIM(){};
};

class DEIM_burgers : public DEIM<fvVectorMatrix>
{
public:
    using DEIM::DEIM;

    PtrList<volVectorField> fieldsA;
    PtrList<volVectorField> fieldsB;
};

/// Structure to implement a newton object for a non-stationary NS problem
struct newton_nmlspg_mdeim_burgers : public newton_argument<double>
{
public:
    newton_nmlspg_mdeim_burgers() {}
    newton_nmlspg_mdeim_burgers(int Nx, int Ny, Burgers &problem, EmbeddingMDEIM &embedding, volVectorField &field) : newton_argument<double>(Nx, Ny),
                                                                                                                      problem(&problem),
                                                                                                                      N_BC(problem.inletIndex.rows()),
                                                                                                                      embedding(&embedding)
    {

        if (problem.liftfield.size() != 0)
        {
            Nphi_u = problem.NUmodes + problem.liftfield.size();
        }
        else
        {
            Nphi_u = problem.NUmodes + 1;
        }
    }

    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const;
    void localAB2generalizedCoord(Eigen::VectorXd matrixBindices);
    Eigen::MatrixXd projectMatrix(Eigen::MatrixXd full_matrix) const;
    Eigen::VectorXd projectVector(Eigen::VectorXd full_vector) const;
    Eigen::VectorXd embedding_ref(const scalar mu) const;

    fvVectorMatrix evaluate_expression(volVectorField &U, const volVectorField &U_old, double mu) const;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> onlineCoeffsAB(double mu,PtrList<volVectorField> listA, PtrList<volVectorField> listB) const;
    Eigen::MatrixXd onlineCoeffsA(double mu, PtrList<volVectorField> listA) const;
    Eigen::MatrixXd onlineCoeffsB(double mu, PtrList<volVectorField> listB) const;
    void init_old(Eigen::VectorXd& U_old);
    void rec_field(Eigen::VectorXd& vec, volVectorField& fieldRec, word name);
    Eigen::VectorXd restrict_decoder();

    Burgers *problem;
    int Nphi_u;
    int N_BC;
    int counterRec{0};
    dimensionedScalar nu;
    scalar dt;
    scalar mu;

    EmbeddingMDEIM *embedding;
    Eigen::VectorXd g_old;

    Eigen::VectorXd y_old;
    Eigen::VectorXd yOldOld;
    Eigen::VectorXd BC;
    Eigen::MatrixXd tauU;

    autoPtr<DEIM_burgers> mdeim;
    PtrList<volVectorField> listAold;
    PtrList<volVectorField> listBold;
    Eigen::VectorXd matrixBindices;
    int total_submeshes_points;
    Eigen::MatrixXd pinvAB; // the shape is n_deim x n_mdeim
    Eigen::MatrixXd pinvBA; // the shape is n_mdeim x n_deim
    Eigen::VectorXd U0_mdeim;
    int total_mp; // half decoder hyperreduced output length
};

/*---------------------------------------------------------------------------*\
                        Class reducedProblem Declaration
\*---------------------------------------------------------------------------*/

/// Class where it is implemented a reduced problem for the <b> steady Navier-stokes </b> problem.
/** In this class are implemented the methods to solve a reduced version of the stationary Navier-stokes
equations. This class must be used together with the steadyNS class */
class NMLSPGMDEIMBurgers : public reducedProblem
{
private:
public:
    // Constructors
    /// Construct Null
    NMLSPGMDEIMBurgers();

    /// Construct Null
    ///
    /// @param[in]  problem  a full order steadyNS problem
    /// @param[in]  tipo     Type of pressure stabilisation method you want to use "SUP" for supremizer, "PPE" for pressure Poisson equation.
    ///
    explicit NMLSPGMDEIMBurgers(Burgers &problem, fileName decoder_path,
                                int latent_dim, Eigen::MatrixXd lat_init,
                                Eigen::VectorXd matrixBindices_,
                                Eigen::VectorXd matrixBsubmeshes_,
                                 int total_mp_);

    ~NMLSPGMDEIMBurgers(){};

    /// parameters to be read from the ITHACAdict file
    ITHACAparameters *para;

    /// DecoderMDEIM
    autoPtr<EmbeddingMDEIM> embedding_mdeim;

    /// Decoder
    autoPtr<Embedding> embedding;

    /// Online inlet velocity vector
    Eigen::MatrixXd vel_now;

    /// Reduced matrix for tangent forces
    Eigen::MatrixXd fTau;

    /// Reduced matrix for normal forces
    Eigen::MatrixXd fN;

    /// Vector to store the previous solution during the Newton procedure
    Eigen::VectorXd y_old;

    /// Vector to store the solution during the Newton procedure
    Eigen::VectorXd y;

    /// Reduced viscosity in case of parametrized viscosity
    dimensionedScalar nu;

    /// List of Eigen matrices to store the online solution
    List<Eigen::MatrixXd> online_solution;

    /// List of pointers to store the modes for velocity
    PtrList<volVectorField> Umodes;

    /// List of pointers to store the snapshots for velocity
    PtrList<volVectorField> Usnapshots;

    /// Recontructed velocity field
    PtrList<volVectorField> UREC;

    /// Recontructed velocity fields list
    PtrList<volVectorField> uRecFields;

    /// Latent velocity fields list
    Eigen::MatrixXd uRecLatent;

    /// List of pointers to store the snapshots for residuals
    PtrList<volScalarField> residualsList;

    /// List of pointers to store the MDEIM matrix and source term
    PtrList<fvVectorMatrix> matrixMDEIMList;

    /// Newton object used to solve the mdeim non linear problem
    newton_nmlspg_mdeim_burgers newton_object_mdeim;

    /// Pointer to the FOM problem
    Burgers *problem;

    /// Latent space dimension
    int Nphi_u;

    /// MDEIM dimensions
    int NUmodes;
    int NmodesDEIMA;
    int NmodesDEIMB;
    Eigen::VectorXd matrixBindices;
    Eigen::VectorXd matrixBsubmeshes;
    int total_mp;

    /// Number of parametrized boundary conditions
    int N_BC;

    /// Counter to count the online solutions
    int count_online_solve = 1;

    /// Penalty Factor
    Eigen::MatrixXd tauU;

    /// Scalar to store the current time
    scalar time;

    /// Scalar to store the time increment
    double dt;

    /// Scalar to store the final time if the online simulation
    scalar finalTime;

    /// Scalar to store the initial time if the online simulation
    scalar tstart;

    /// A variable for storing the reduced coefficients
    double storeEvery;

    /// A variable for exporting the fields
    double exportEvery;

    /// A variable for exporting the sList
    double exportResidual{0};

    /// A variable for exporting the sList
    double exportMatricesMDEIM{0};

    /// A variable for starting solving the reduced system from zero or not
    bool startFromZero = false;

    /// Number of timesteps calculated for the iterative penalty method
    int timeStepPenalty = 1;

    /// Penalty Factor determined with iterative solver
    Eigen::MatrixXd tauIter;

    /// Maximum number of iterations to be done for the computation of the penalty factor
    scalar maxIterPenalty;

    /// Tolerance for the sList of the boundary values, there is the same tolerance for velocity and temperature
    scalar tolerancePenalty;

    // Functions

    /// Method to determine the penalty factors iteratively.
    ///
    /// @param[in]  vel_now   The vector of online velocity. It is defined in
    /// with an Eigen::MatrixXd and must have one col or as many cols as timesteps to be performed
    /// in case of time-dependent BCs and as many rows as the number of parametrized boundary conditions.
    ///
    /// @param[in]  tauIter   The penalty factors determined one iteration step
    ///
    /// @param[in]  startSnap The first snapshot taken from the offline snapshots
    /// and used to get the reduced initial condition.
    ///
    Eigen::MatrixXd penalty(Eigen::MatrixXd &vel_now, Eigen::MatrixXd &tauIter,
                            int startSnap = 0);

    /// Method to perform an online solve using a supremizer stabilisation method
    ///
    /// @param[in]  vel_now   The vector of online velocity. It is defined in
    /// with an Eigen::MatrixXd and must have one col and as many rows as the number
    /// of parametrized boundary conditions.
    /// @param[in]  startSnap The first snapshot taken from the offline snapshots
    /// and used to get the reduced initial condition.
    ///
    //void solveOnline(Eigen::MatrixXd vel_now, int startSnap = 0);

    /// Method to perform an online solve using a supremizer stabilisation method
    ///
    /// @param[in]  vel_now   The vector of online velocity. It is defined in
    /// with an Eigen::MatrixXd and must have one col and as many rows as the number
    /// of parametrized boundary conditions.
    /// @param[in]  startSnap The first snapshot taken from the offline snapshots
    /// and used to get the reduced initial condition.
    ///
    void solveOnline(int startSnap = 0);

    /// Method to perform an online solve using a supremizer stabilisation method
    ///
    /// @param[in]  vel_now   The vector of online velocity. It is defined in
    /// with an Eigen::MatrixXd and must have one col and as many rows as the number
    /// of parametrized boundary conditions.
    /// @param[in]  startSnap The first snapshot taken from the offline snapshots
    /// and used to get the reduced initial condition.
    ///
    void solveOnline(Eigen::MatrixXd mu, int startSnap = 0);

    ///
    /// Method to reconstruct the solutions from an online solve.
    ///
    /// @param[in]  exportFields  A boolean variable which determines whether to export fields or not
    /// @param[in]  folder        The folder where to output the solutions in case on wants to
    /// @param[in]  printevery    The printevery variabel, when it is 1 the function outputs all stored fields
    ///
    void reconstructNM(fileName decoder_path, Burgers &problem, bool exportFields = false,
                     fileName folder = "./ITHACAoutput/online_rec");

    ///
    /// @brief      Sets the online velocity.
    ///
    /// @param[in]  vel   The velocity
    ///
    /// @return     The rescaled velocity vector
    ///
    Eigen::MatrixXd setOnlineVelocity(Eigen::MatrixXd vel);

    ///
    /// @brief      Projects and reconstructs the snapshots in Umodes.
    ///
    /// @param[in]  folder     The folder where to output the solution
    ///
    void trueProjection(fileName folder);

    void NMLSPG_MMDEIM(int NmodesDEIMA, int NmodesDEIMB);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
