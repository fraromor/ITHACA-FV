
Info<< "Reading field U\n" << endl;

volVectorField U
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
);


Info<< "Reading transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);


Info<< "Reading viscosity nu\n" << endl;

_nu = autoPtr<dimensionedScalar>(new dimensionedScalar("nu", dimViscosity, transportProperties));
dimensionedScalar& nu = _nu();

Info << "####################DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/createFields.H, line 72####################" << endl;
//CHECK_end

#include "createPhi.H"
