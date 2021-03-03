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

#include "compressibleNS.H"
#include "Foam2Eigen.H"
/// \file
/// Source file of the compressibleNS class.

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Construct Null
compressibleNS::compressibleNS() {}

// Construct from zero
compressibleNS::compressibleNS(int argc, char* argv[])
{
    _args = autoPtr<argList>
            (
                new argList(argc, argv)
            );

    if (!_args->checkRootCase())
    {
        Foam::FatalError.exit();
    }

    argList& args = _args();
    #include "createTime.H"
    #include "createMesh.H"
    _pimple = autoPtr<pimpleControl>
              (
                  new pimpleControl
                  (
                      mesh
                  )
              );
    ITHACAdict = new IOdictionary
    (
        IOobject
        (
            "ITHACAdict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    #include "createControl.H" // TODO CHECK
    #include "createFields.H"
    #include "initContinuityErrs.H"

    turbulence->validate();

    para = ITHACAparameters::getInstance(mesh, runTime);

    bcMethod = ITHACAdict->lookupOrDefault<word>("bcMethod", "lift");
    M_Assert(bcMethod == "lift" || bcMethod == "penalty",
             "The BC method must be set to lift or penalty in ITHACAdict");

    // timedepbcMethod = ITHACAdict->lookupOrDefault<word>("timedepbcMethod", "no");
    // M_Assert(timedepbcMethod == "yes" || timedepbcMethod == "no",
    //          "The BC method can be set to yes or no");

    // // TODO check on ITHACADict tutorial
    // timeDerivativeSchemeOrder =
    //     ITHACAdict->lookupOrDefault<word>("timeDerivativeSchemeOrder", "second");
    // M_Assert(timeDerivativeSchemeOrder == "first"
    //          || timeDerivativeSchemeOrder == "second",
    //          "The time derivative approximation must be set to either first or second order scheme in ITHACAdict");

    offline = ITHACAutilities::check_off();
    // podex = ITHACAutilities::check_pod();
    // supex = ITHACAutilities::check_sup();
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void compressibleNS::truthSolve(List<scalar> mu_now, fileName folder)
{
    Time& runTime = _runTime();
    fvMesh& mesh = _mesh();
    fv::options& fvOptions = _fvOptions();
    pimpleControl& pimple = _pimple();
    psiThermo& thermo = _thermo();
    surfaceScalarField& phi = _phi();
    volScalarField& rho = _rho();
    volScalarField& p = _p();
    volVectorField& U = _U();
    volScalarField& psi = _psi();
    volScalarField& e = _E();
    volScalarField& K = _K();
    IOMRFZoneList& MRF = _MRF();
    // compressible::turbulenceModel& turbulence = _turbulence();
    instantList Times = runTime.times();
    runTime.setEndTime(finalTime);

    // Perform a TruthSolve
    Info << " # DEBUG compressibleNS.C, line 135 # " << Times << " " << timeStep << endl;
    runTime.setTime(Times[1], 1);
    runTime.setDeltaT(timeStep);
    nextWrite = startTime;
    Info << " # DEBUG compressibleNS.C, line 139 # " << runTime.timeName() << nl << endl;

    // Set time-dependent velocity BCs for initial condition
    if (timedepbcMethod == "yes")
    {
        for (label i = 0; i < inletPatch.rows(); i++)
        {
            Vector<double> inl(0, 0, 0);

            for (label j = 0; j < inl.size(); j++)
            {
                inl[j] = timeBCoff(i * inl.size() + j, 0);
            }

            assignBC(U, inletPatch(i, 0), inl);
        }
    }
    Info << " # DEBUG compressibleNS.C, line 143 # " << endl;

    // Export and store the initial conditions for velocity and pressure
    ITHACAstream::exportSolution(U, name(counter), folder);
    ITHACAstream::exportSolution(rho, name(counter), folder);
    ITHACAstream::exportSolution(e, name(counter), folder);
    std::ofstream of(folder + name(counter) + "/" + runTime.timeName());
    Ufield.append(U.clone());
    rhofield.append(rho.clone());
    efield.append(e.clone());

    // #include "readTimeControls.H"
    runTime.setEndTime(finalTime);

    Info<< "deltaT = " <<  runTime.deltaTValue() << endl;

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.loop())
    {
        Info << "Time = " << runTime.timeName() << nl << endl;

        #include "compressibleCourantNo.H"

        // #include "setDeltaT.H"

        // Set time-dependent velocity BCs
        // if (timedepbcMethod == "yes")
        // {
        //     for (label i = 0; i < inletPatch.rows(); i++)
        //     {
        //         Vector<double> inl(0, 0, 0);

        //         for (label j = 0; j < inl.size(); j++)
        //         {
        //             inl[j] = timeBCoff(i * inl.size() + j, counter2);
        //         }

        //         assignBC(U, inletPatch(i, 0), inl);
        //     }

        //     counter2 ++;
        // }

        #include "rhoEqn.H"

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            #include "UEqn.H"
            #include "EEqn.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                turbulence->correct();
            }
        }

        rho = thermo.rho();

        runTime.write();

        // Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        //      << "  ClockTime = " << runTime.elapsedClockTime() << " s"
        //      << nl << endl;

        runTime.printExecutionTime(Info);

        if (checkWrite(runTime))
        {
            ITHACAstream::exportSolution(U, name(counter), folder);
            ITHACAstream::exportSolution(rho, name(counter), folder);
            ITHACAstream::exportSolution(e, name(counter), folder);
            std::ofstream of(folder + name(counter) + "/" +
                             runTime.timeName());
            Ufield.append(U.clone());
            rhofield.append(rho.clone());
            efield.append(e.clone());
            counter++;
            nextWrite += writeEvery;
            writeMu(mu_now);

            // --- Fill in the mu_samples with parameters (time, mu) to be used for the PODI sample points
            mu_samples.conservativeResize(mu_samples.rows() + 1, mu_now.size() + 1);
            mu_samples(mu_samples.rows() - 1, 0) = atof(runTime.timeName().c_str());

            for (label i = 0; i < mu_now.size(); i++)
            {
                mu_samples(mu_samples.rows() - 1, i + 1) = mu_now[i];
            }
        }
    }

    // Resize to Unitary if not initialized by user (i.e. non-parametric problem)
    if (mu.cols() == 0)
    {
        mu.resize(1, 1);
    }

    if (mu_samples.rows() == counter * mu.cols())
    {
        ITHACAstream::exportMatrix(mu_samples, "mu_samples", "eigen",
                                   folder);
    }
}


bool compressibleNS::checkWrite(Time& timeObject)
{
    scalar diffnow = mag(nextWrite - atof(timeObject.timeName().c_str()));
    scalar diffnext = mag(nextWrite - atof(timeObject.timeName().c_str()) -
                          timeObject.deltaTValue());

    if ( diffnow < diffnext)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void compressibleNS::change_initial_velocity(double mu)
{
    Info << " # DEBUG compressibleNS.C, line 269 # " << endl;
    Vector<double> value(mu, 0, 0);
    volVectorField& U = _U();
    Info << " # DEBUG compressibleNS.C, line 272 # " << endl;
    this->assignIF(U, value);
    this->assignBC(U, 0, value);
    Info << " # DEBUG compressibleNS.C, line 275 # " << endl;
    // ITHACAstream::exportSolution(U, name(mu), "./initial_data/");
    // ITHACAstream::exportSolution(_U(), name(mu), "./initial_data_/");
    Info << " # DEBUG compressibleNS.C, line 278 # " << endl;
}