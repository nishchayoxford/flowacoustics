// SPDX-License-Identifier: GPL-3.0-or-later
// Read-only collective post-PIMPLE guard for installed OpenCFD v2206.
// Courant formula matches CourantNo.H and the verified runtimeMetrics.H.
#include "PstreamReduceOps.H"
#include "IOmanip.H"
#include <cmath>
#include <fstream>
#include <iomanip>

template<class GeoField>
bool finiteProductionField(const GeoField& field)
{
    using namespace Foam;
    for (const auto& value : field.primitiveField())
        if (!std::isfinite(mag(value))) return false;
    for (const auto& patch : field.boundaryField())
        for (const auto& value : patch)
            if (!std::isfinite(mag(value))) return false;
    return true;
}

void productionGuard(const Foam::fvMesh& mesh, const Foam::volVectorField& U,
                     const Foam::surfaceScalarField& phi)
{
    using namespace Foam;
    bool finite = finiteProductionField(U) && finiteProductionField(phi);
    for (const word& name : wordList({"p", "nut", "k", "omega"}))
        if (mesh.foundObject<volScalarField>(name))
            finite = finiteProductionField(mesh.lookupObject<volScalarField>(name)) && finite;
    const scalarField& V = mesh.V().field();
    for (scalar volume : V)
        finite = finite && std::isfinite(volume) && volume > 0;
    reduce(finite, andOp<bool>());
    if (!finite)
        FatalErrorInFunction << "Nonfinite production field/volume" << exit(FatalError);

    surfaceScalarField effective(fvc::meshPhi(U));
    const surfaceScalarField* fluxes[] = {&phi, &mesh.phi(), &effective};
    scalar maxima[3] = {0,0,0};
    for (label k=0; k<3; ++k)
    {
        finite = finiteProductionField(*fluxes[k]) && finite;
        scalarField sums(fvc::surfaceSum(mag(*fluxes[k]))().primitiveField());
        forAll(sums, i)
        {
            const scalar co = 0.5*mesh.time().deltaTValue()*sums[i]/V[i];
            finite = finite && std::isfinite(co);
            maxima[k] = max(maxima[k], co);
        }
        reduce(maxima[k], maxOp<scalar>());
    }
    reduce(finite, andOp<bool>());
    Info<< setprecision(17);
    Info<< "MRL2D_POST time=" << mesh.time().value()
        << " fluid=" << maxima[0] << " meshRaw=" << maxima[1]
        << " meshEffective=" << maxima[2] << " finite=" << (finite ? 1 : 0) << endl;
    if (!finite || maxima[0]>1 || maxima[1]>1 || maxima[2]>1)
        FatalErrorInFunction << "Post-solve production Courant/nonfinite gate" << exit(FatalError);
}
