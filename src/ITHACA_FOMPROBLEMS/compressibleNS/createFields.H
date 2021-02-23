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
Info<< "Reading thermophysical properties\n" << endl;

autoPtr<psiThermo> pThermo
(
    psiThermo::New(mesh)
);

// TODO create initial _pThermo0
psiThermo& thermo = pThermo();
thermo.validate(args.executable(), "e");

volScalarField& p = thermo.p();

// TODO create autoPtr rho and _rho0
volScalarField rho
(
    IOobject
    (
        "rho",
        runTime.timeName(),
        mesh
    ),
    thermo.rho()
);

// _p = autoPtr<volScalarField>
//      (
//          new volScalarField
//          (
//              IOobject
//              (
//                  "p",
//                  runTime.timeName(),
//                  mesh,
//                  IOobject::MUST_READ,
//                  IOobject::AUTO_WRITE
//              ),
//              mesh
//          )
//      );
// volScalarField& p = _p();

// volScalarField p0(p);

// _p0 = autoPtr<volScalarField>
//       (
//           new volScalarField(p0)
//       );

Info << "Reading field U\n" << endl;
_U = autoPtr<volVectorField>
     (
         new volVectorField
         (
             IOobject
             (
                 "U",
                 runTime.timeName(),
                 mesh,
                 IOobject::MUST_READ,
                 IOobject::AUTO_WRITE
             ),
             mesh
         )
     );
volVectorField& U = _U();

volVectorField U0(U);

_U0 = autoPtr<volVectorField>
      (
          new volVectorField(U0)
      );

#include "compressibleCreatePhi.H"

mesh.setFluxRequired(p.name());

Info<< "Creating turbulence model\n" << endl;
autoPtr<compressible::turbulenceModel> turbulence
(
    compressible::turbulenceModel::New
    (
        rho,
        U,
        phi,
        thermo
    )
);

#include "createK.H"

_MRF = autoPtr<IOMRFZoneList>
(
    new IOMRFZoneList(mesh)
);

#include "createFvOptions.H"

// label pRefCell = 0;
// scalar pRefValue = 0.0;
// setRefCell(p, mesh.solutionDict().subDict("SIMPLE"), pRefCell, pRefValue);


// _laminarTransport = autoPtr<singlePhaseTransportModel>
//                     (
//                         new singlePhaseTransportModel( U, phi )
//                     );
// singlePhaseTransportModel& laminarTransport = _laminarTransport();

// turbulence = autoPtr<incompressible::turbulenceModel>
//              (
//                  incompressible::turbulenceModel::New(U, phi, laminarTransport)
//              );
