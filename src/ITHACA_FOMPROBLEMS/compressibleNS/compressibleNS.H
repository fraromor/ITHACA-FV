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
    compressibleNS
Description
    Reduction class for a NON-Stationary NS problem
SourceFiles
    compressibleNS.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the compressibleNS class.
/// \dir
/// Directory containing the header and source files for the compressibleNS class.

#ifndef compressibleNS_H
#define compressibleNS_H
#include "fvCFD.H"
#include "psiThermo.H"
#include "turbulentFluidThermoModel.H"
#include "pimpleControl.H"
#include "fvOptions.H"

#include "IOMRFZoneList.H" // ? necessary

#include "steadyNS.H"

#include <iostream>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

/*---------------------------------------------------------------------------*\
                        Class compressibleNS Declaration
\*---------------------------------------------------------------------------*/

/// Implementation of a parametrized full order <b> compressible NS problem </b> and preparation of the the reduced matrices for the online solve.
/** In this class are implemented the methods for the offline solve of a compressible NS problem
and the for the generation of the reduced matrices for subsequent online solve, this class is a son
of the compressibelNS class */
class compressibleNS: public steadyNS
{
    public:
        // Constructors
        /// Construct Null
        compressibleNS();

        /// Construct with argc and argv
        compressibleNS(int argc, char* argv[]);

        // Dummy variables to transform pimplefoam into a class
        /// pimpleControl
        autoPtr<pimpleControl> _pimple;

        /// Turbulence model
        autoPtr<compressible::turbulenceModel> turbulence;

        /// Thermodynamic properties
        autoPtr<psiThermo> _thermo;

        /// Energy field
        autoPtr<volScalarField> _E;

        /// Compressibility field
        autoPtr<volScalarField> _rho;

        /// Compressibility field
        autoPtr<volScalarField> _psi;

        /// Kinetic Energy field
        autoPtr<volScalarField> _K;


        /// List of pointers to store initial solutions
        /// Initial thermodynamic properties (for restart purposes)
        autoPtr<psiThermo> _thermo0;

        /// Initial pressure (for restart purposes)
        autoPtr<volScalarField> _p0;

        /// Initial Energy field (for restart purposes)
        autoPtr<volScalarField> _E0;

        /// Initial Velocity field (for restart purposes)
        autoPtr<volVectorField> _U0;

        /// Initial Density field (for restart purposes)
        autoPtr<volScalarField> _rho0;

        /// Initial Flux field (for restart purposes)
        autoPtr<surfaceScalarField> _phi0;

        /// Initial Compressibility field (for restart purposes)
        autoPtr<volScalarField> _psi0;

        /// Initial Kinetic Energy field (for restart purposes)
        autoPtr<volScalarField> _K0;

        /// Member fields
        /// List of pointers used to form the pressure snapshots matrix
        PtrList<volScalarField> rhofield;

        /// List of pointers used to form the pressure snapshots matrix
        PtrList<volScalarField> efield;

        /// TODO remove
        PtrList<volScalarField> saveField;

        /// List of pointers used to form the density modes
        volScalarModes rhomodes;

        /// List of pointers used to form the internal energy modes
        volScalarModes emodes;


        // Specific variable for the unstationary case
        /// Start Time (initial time to start storing the snapshots)
        scalar startTime;

        /// Final time (final time of the simulation and consequently of the acquisition of the snapshots)
        scalar finalTime;

        /// Time step of the simulation
        scalar timeStep;

        /// Time step of the writing procedure
        scalar writeEvery = timeStep;

        /// Auxiliary variable to store the next writing instant
        scalar nextWrite;

        /// adjustTimeStep
        bool adjustTimeStep;

        /// maxCourant
        scalar maxCo;

        /// maxDeltaT
        scalar maxDeltaT;

        // Counter used for the time-dependent boundary conditions
        label counter2 = 1;

        /// Time-dependent Boundary Method
        word timedepbcMethod;

        // Matrix containing the time-dependent boundary conditions for each time step
        Eigen::MatrixXd timeBCoff;

        // Time derivative numerical scheme order
        word timeDerivativeSchemeOrder;

        // Single truth solve counter
        int counter_truth = 0;
        // Functions

        //--------------------------------------------------------------------------
        /// @brief      Perform a truthsolve
        /// @param[in]  mu_now  The actual value of the parameter for this truthSolve. Used only
        /// to construct mu_interp matrix which is written out in a specified folder, also for par
        /// file in the Parameters folder.
        ///
        void truthSolve(List<scalar> mu_now,
                        fileName folder = "./ITHACAoutput/Offline/");

        //--------------------------------------------------------------------------
        /// Function to check if the solution must be exported.
        ///
        /// @param      timeObject  The time object of OpenFOAM.
        ///
        /// @return     1 if we must write 0 elsewhere.
        ///
        bool checkWrite(Time& timeObject);

        //--------------------------------------------------------------------------
        /// Function to change the first component of the initial velocity
        ///
        /// @param[in]  mu    first component of velocity (scalar)
        ///
        void change_initial_velocity(double mu);

        void restart();
};
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif










