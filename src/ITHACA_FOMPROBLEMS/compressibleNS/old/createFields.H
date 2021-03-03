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
Info << "Reading thermophysical properties\n"
     << endl;

_thermo = autoPtr<psiThermo>(psiThermo::New(mesh));

// ? create initial _pThermo0
_thermo->validate(args.executable(), "e");

_p = autoPtr<volScalarField>(new volScalarField(_thermo->p()));
_p0 = autoPtr<volScalarField>(new volScalarField(_p()));

_rho = autoPtr<volScalarField>(
    new volScalarField(
        IOobject(
            "rho",
            runTime.timeName(),
            mesh),
        _thermo->rho())
    );

_rho0 = autoPtr<volScalarField>(new volScalarField(_rho()));

_E = autoPtr<volScalarField>(new volScalarField(_thermo->he()));
_E0 = autoPtr<volScalarField>(new volScalarField(_E()));

_psi = autoPtr<volScalarField>(new volScalarField(_thermo->psi()));
_psi0 = autoPtr<volScalarField>(new volScalarField(_psi()));

Info << "Reading field U\n" << endl;

_U = autoPtr<volVectorField>(
    new volVectorField(
        IOobject(
            "U",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));

_U0 = autoPtr<volVectorField>(
    new volVectorField(_U()));

Info<< "Reading/calculating face flux field phi\n" << endl;

_phi = autoPtr<surfaceScalarField>(new surfaceScalarField
(
    IOobject
    (
        "phi",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    linearInterpolate(_rho()*_U()) & mesh.Sf()
));


mesh.setFluxRequired(_p().name());

Info << "Creating turbulence model\n" << endl;

turbulence = autoPtr<compressible::turbulenceModel>(
    compressible::turbulenceModel::New(
        _rho(),
        _U(),
        _phi(),
        _thermo()));

Info << "Creating field kinetic energy K\n"
     << endl;

_K = autoPtr<volScalarField>(new volScalarField("K", 0.5 * magSqr(_U())));
_K0 = autoPtr<volScalarField>(new volScalarField(_K()));

if (_U().nOldTimes())
{
    volVectorField *Uold = &_U().oldTime();
    volScalarField *Kold = &_K().oldTime();
    *Kold == 0.5 * magSqr(*Uold);

    while (Uold->nOldTimes())
    {
        Uold = &Uold->oldTime();
        Kold = &Kold->oldTime();
        *Kold == 0.5 * magSqr(*Uold);
    }
}

_MRF = autoPtr<IOMRFZoneList>(new IOMRFZoneList(mesh));

_fvOptions = autoPtr<fv::options>(new fv::options(mesh));
