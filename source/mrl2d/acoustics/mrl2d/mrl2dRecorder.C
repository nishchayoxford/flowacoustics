// SPDX-License-Identifier: GPL-3.0-or-later
// OpenCFD v2206, strict-2D native field recorder. No acoustic solver is implied.
#include "fvMeshFunctionObject.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "fvMesh.H"
#include "Time.H"
#include "OSspecific.H"
#include "labelIOList.H"
#include "interpolationCellPoint.H"
#include "PstreamReduceOps.H"
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>

namespace Foam { namespace functionObjects {
class mrl2dRecorder : public fvMeshFunctionObject
{
    scalar depth_, rho_, omega_, z_;
    bool envelope_;
    label segments_;
    wordList bodies_;
    std::ofstream stream_;
    autoPtr<labelIOList> faceAddress_;
    bool initialized_ = false;
    scalar lastTime_ = -GREAT;
    struct Sample {point x; vector normal; scalar length; label cell; point cellCentre;};
    std::vector<Sample> envelopePoints_;

    void init()
    {
        const std::uint16_t endian = 1;
        if (sizeof(double) != 8 || *reinterpret_cast<const char*>(&endian) != 1)
            FatalErrorInFunction << "Recorder requires little-endian IEEE binary64" << exit(FatalError);
        if (Pstream::parRun())
            faceAddress_.reset(new labelIOList(IOobject
            (
                "faceProcAddressing", mesh_.facesInstance(), polyMesh::meshSubDir,
                mesh_, IOobject::MUST_READ, IOobject::NO_WRITE
            )));
        const fileName dir = mesh_.time().path()/"postProcessing"/"mrl2dSources";
        mkDir(dir);
        const fileName path = dir/("rank" + Foam::name(Pstream::myProcNo()) + ".bin");
        if (isFile(path))
            FatalErrorInFunction << "Refusing existing source file " << path << exit(FatalError);
        stream_.open(path.c_str(), std::ios::binary | std::ios::app);
        if (!stream_) FatalErrorInFunction << "Cannot open " << path << exit(FatalError);
        if (envelope_)
        {
            const scalar x0=-0.14, x1=0.30, y0=-0.14, y1=0.14;
            for (label side=0; side<4; ++side)
                for (label j=0; j<segments_; ++j)
                {
                    const scalar f=(j+0.5)/segments_;
                    Sample s;
                    if (side==0) {s.x=point(x0+(x1-x0)*f,y0,z_); s.normal=vector(0,-1,0);}
                    if (side==1) {s.x=point(x1,y0+(y1-y0)*f,z_); s.normal=vector(1,0,0);}
                    if (side==2) {s.x=point(x1-(x1-x0)*f,y1,z_); s.normal=vector(0,1,0);}
                    if (side==3) {s.x=point(x0,y1-(y1-y0)*f,z_); s.normal=vector(-1,0,0);}
                    s.length=(side%2 ? y1-y0 : x1-x0)/segments_;
                    s.cell=mesh_.findCell(s.x);
                    if (s.cell>=0)
                    {
                        for (const word& zoneName : wordList({"carrierZone", "blade0Zone", "blade1Zone", "blade2Zone"}))
                        {
                            const label zi=mesh_.cellZones().findZoneID(zoneName);
                            if (zi>=0 && mesh_.cellZones()[zi].whichCell(s.cell)>=0)
                                FatalErrorInFunction << "Envelope intersects moving zone at " << s.x << exit(FatalError);
                        }
                    }
                    label owner = s.cell>=0 ? Pstream::myProcNo() : labelMax;
                    reduce(owner, minOp<label>());
                    if (owner==labelMax)
                        FatalErrorInFunction << "No native interpolation cell for envelope " << s.x << exit(FatalError);
                    if (owner != Pstream::myProcNo()) s.cell=-1;
                    if (s.cell>=0) s.cellCentre=mesh_.cellCentres()[s.cell];
                    envelopePoints_.push_back(s);
                }
        }
        initialized_=true;
    }

    void emit(scalar t, label kind, label id, label blade, const point& x,
              const vector& n, scalar ds, scalar p, const vector& u,
              const vector& vs, label cell)
    {
        double row[16]={t,double(kind),double(id),double(blade),x.x(),x.y(),x.z(),
                        n.x(),n.y(),ds,rho_*p,u.x(),u.y(),vs.x(),vs.y(),double(cell)};
        for (double value : row)
            if (!std::isfinite(value))
                FatalErrorInFunction << "Nonfinite native source" << exit(FatalError);
        if (ds<=0 || mag(n.z())>1e-10 || mag(u.z())>1e-8)
            FatalErrorInFunction << "Non-2D source geometry/velocity" << exit(FatalError);
        stream_.write(reinterpret_cast<const char*>(row), sizeof(row));
    }

public:
    TypeName("mrl2dRecorder");
    mrl2dRecorder(const word& name, const Time& runTime, const dictionary& dict)
    : fvMeshFunctionObject(name, runTime, dict),
      depth_(dict.get<scalar>("depth")), rho_(dict.get<scalar>("rho")),
      omega_(dict.get<scalar>("omega")), z_(dict.get<scalar>("zSample")),
      envelope_(dict.get<bool>("envelope")),
      segments_(dict.getOrDefault<label>("segmentsPerSide",80)),
      bodies_(dict.getOrDefault<wordList>("bodyPatches",wordList({"blade0","blade1","blade2"})))
    {
        if (depth_<=0 || rho_<=0 || segments_<1)
            FatalIOErrorInFunction(dict) << "Invalid depth/rho/segments" << exit(FatalIOError);
    }
    bool read(const dictionary& dict) override {return fvMeshFunctionObject::read(dict);}
    bool write() override {return true;}
    bool execute() override
    {
        const scalar t=mesh_.time().value();
        if (mag(t-lastTime_)<SMALL) return true;
        if (t<lastTime_) FatalErrorInFunction << "Nonmonotonic sampling time" << exit(FatalError);
        if (!initialized_) init();
        const auto& p=mesh_.lookupObject<volScalarField>("p");
        const auto& U=mesh_.lookupObject<volVectorField>("U");
        const scalar cs=std::cos(omega_*t), sn=std::sin(omega_*t);
        const point initial[3]={point(0,.055,0),point(std::sqrt(3.0)*.055/2,-.0275,0),
                               point(-std::sqrt(3.0)*.055/2,-.0275,0)};
        label localRows=0;
        for (const word& patchName : bodies_)
        {
            const label b=patchName=="blade0" ? 0 : patchName=="blade1" ? 1 : patchName=="blade2" ? 2 : -1;
            const label pi=mesh_.boundaryMesh().findPatchID(patchName);
            if (b<0 || pi<0) FatalErrorInFunction << "Missing/invalid body patch " << patchName << exit(FatalError);
            const auto& patch=mesh_.boundary()[pi];
            const auto& cf=patch.Cf();
            const auto& sf=patch.Sf();
            const auto& area=patch.magSf();
            const point c(cs*initial[b].x()+sn*initial[b].y(), -sn*initial[b].x()+cs*initial[b].y(),0);
            forAll(patch, j)
            {
                const point x=cf[j];
                const vector vs(omega_*c.y()+omega_/2*(x.y()-c.y()),
                                -omega_*c.x()-omega_/2*(x.x()-c.x()),0);
                const label localFace=patch.start()+j;
                const label id=faceAddress_.valid() ? mag((*faceAddress_)[localFace])-1 : localFace;
                emit(t,0,id,b,x,-sf[j]/area[j],area[j]/depth_,p.boundaryField()[pi][j],
                     U.boundaryField()[pi][j],vs,patch.faceCells()[j]);
                ++localRows;
            }
        }
        if (envelope_)
        {
            interpolationCellPoint<scalar> ip(p);
            interpolationCellPoint<vector> iu(U);
            for (label j=0; j<label(envelopePoints_.size()); ++j)
            {
                const Sample& s=envelopePoints_[j];
                if (s.cell<0) continue;
                if (mag(mesh_.cellCentres()[s.cell]-s.cellCentre)>1e-12)
                    FatalErrorInFunction << "Stationary envelope cell moved" << exit(FatalError);
                emit(t,1,j,-1,s.x,s.normal,s.length,ip.interpolate(s.x,s.cell),
                     iu.interpolate(s.x,s.cell),vector::zero,s.cell);
                ++localRows;
            }
        }
        stream_.flush();
        if (!stream_) FatalErrorInFunction << "Native source write failure" << exit(FatalError);
        lastTime_=t;
        Info<< "MRL2D_SOURCE time=" << t << " rows=" << returnReduce(localRows,sumOp<label>()) << nl;
        return true;
    }
};
defineTypeNameAndDebug(mrl2dRecorder,0);
addToRunTimeSelectionTable(functionObject,mrl2dRecorder,dictionary);
}}
