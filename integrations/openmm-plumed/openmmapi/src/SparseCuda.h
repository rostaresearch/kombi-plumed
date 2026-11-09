// Optional OpenMM 8.1 CUDA transport for PLUMED.  No bias mathematics lives here.
#ifndef PLUMED_SPARSE_CUDA_H
#define PLUMED_SPARSE_CUDA_H
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeKernel.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/ComputeVectorTypes.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaPlatform.h"
#include <algorithm>
#include <memory>

namespace PlumedPlugin {
class SparseCuda {
    class Before : public OpenMM::ComputeContext::ForcePreComputation {
        SparseCuda& s;
    public:
        Before(SparseCuda& s) : s(s) {}
        void computeForceAndEnergy(bool, bool, int groups) {
            if ((groups & (1 << s.group)) != 0) {
                OpenMM::ContextSelector select(s.cc);
                s.energy = s.impl.computeForce(s.context, s.positions, s.forces);
            }
        }
    };
    class After : public OpenMM::ComputeContext::ForcePostComputation {
        SparseCuda& s;
    public:
        After(SparseCuda& s) : s(s) {}
        double computeForceAndEnergy(bool includeForces, bool, int groups) {
            if ((groups & (1 << s.group)) == 0) return 0;
            OpenMM::ContextSelector select(s.cc);
            if (includeForces) s.scatter();
            return s.energy;
        }
    };
    class Reordered : public OpenMM::ComputeContext::ReorderListener {
        std::shared_ptr<bool> dirty;
    public:
        Reordered(std::shared_ptr<bool> dirty) : dirty(dirty) {}
        void execute() { *dirty = true; }
    };
    OpenMM::ComputeContext& cc;
    OpenMM::ContextImpl& context;
    PlumedForceImpl& impl;
    int group;
    double energy = 0;
    std::shared_ptr<bool> dirty;
    std::vector<int> requested, deviceIndex, inverse;
    OpenMM::ComputeArray indices, packedPos, packedCorrection, packedForce;
    OpenMM::ComputeKernel gatherKernel, scatterKernel;
    std::vector<OpenMM::mm_float4> fp, fc;
    std::vector<OpenMM::mm_double4> dp;
    std::vector<float> ff;
    std::vector<double> df;
    bool initialized = false;
public:
    std::vector<OpenMM::Vec3> positions, forces;
    SparseCuda(OpenMM::ComputeContext& cc, OpenMM::ContextImpl& context,
               PlumedForceImpl& impl, int group) : cc(cc), context(context),
               impl(impl), group(group), dirty(new bool(true)),
               positions(cc.getNumAtoms()), forces(cc.getNumAtoms()) {
        cc.addPreComputation(new Before(*this));
        cc.addPostComputation(new After(*this));
        cc.addReorderListener(new Reordered(dirty));
    }
    void gather(plumed p) {
        // PLUMED owns dependency selection, including its full first-step list.
        int count = 0;
        plumed_cmd(p, "createFullList", &count);
        int* list = nullptr;
        plumed_cmd(p, "getFullList", &list);
        std::vector<int> next;
        if (count > 0) next.assign(list, list+count);
        plumed_cmd(p, "clearFullList", nullptr);
        if (count < 0 || count > cc.getNumAtoms())
            throw OpenMM::OpenMMException("Invalid PLUMED sparse atom list");
        if (next != requested) { requested.swap(next); *dirty = true; }
        if (!initialized) initializeBuffers();
        const int capacity = std::max(count, 1);
        if (indices.getSize() != capacity) {
            indices.resize(capacity); packedPos.resize(capacity);
            if (cc.getUseMixedPrecision()) packedCorrection.resize(capacity);
            packedForce.resize(3*capacity);
            *dirty = true;
        }
        if (*dirty) {
            inverse.resize(cc.getNumAtoms());
            const auto& order = cc.getAtomIndex();
            for (int i = 0; i < cc.getNumAtoms(); i++) inverse[order[i]] = i;
            deviceIndex.assign(capacity, 0);
            int previous = -1;
            for (int i = 0; i < count; i++) {
                // PLUMED's full list is sorted and unique; reject invalid mappings.
                if (requested[i] <= previous || requested[i] >= cc.getNumAtoms())
                    throw OpenMM::OpenMMException("Invalid PLUMED sparse atom mapping");
                previous = requested[i];
                deviceIndex[i] = inverse[requested[i]];
            }
            indices.upload(deviceIndex);
            *dirty = false;
        }
        if (count == 0) return;
        // Refresh array arguments after resize.  Download raw storage, then reconstruct
        // on the host in the same order as CudaUpdateStateDataKernel::getPositions().
        gatherKernel->setArg(0, count);
        gatherKernel->setArg(1, indices);
        gatherKernel->setArg(2, cc.getPosq());
        gatherKernel->setArg(3, packedPos);
        if (cc.getUseMixedPrecision()) {
            gatherKernel->setArg(4, cc.getPosqCorrection());
            gatherKernel->setArg(5, packedCorrection);
        }
        gatherKernel->execute(count);
        if (cc.getUseDoublePrecision()) packedPos.download(dp);
        else packedPos.download(fp);
        if (cc.getUseMixedPrecision()) packedCorrection.download(fc);
        OpenMM::Vec3 a, b, c;
        cc.getPeriodicBoxVectors(a, b, c);
        const auto& offsets = cc.getPosCellOffsets();
        for (int i = 0; i < count; i++) {
            OpenMM::Vec3 v;
            if (cc.getUseDoublePrecision()) v = OpenMM::Vec3(dp[i].x, dp[i].y, dp[i].z);
            else if (cc.getUseMixedPrecision())
                v = OpenMM::Vec3((double)fp[i].x+(double)fc[i].x,
                                (double)fp[i].y+(double)fc[i].y,
                                (double)fp[i].z+(double)fc[i].z);
            else v = OpenMM::Vec3(fp[i].x, fp[i].y, fp[i].z);
            const auto& o = offsets[deviceIndex[i]];
            positions[requested[i]] = v-a*o.x-b*o.y-c*o.z;
        }
    }
private:
    void initializeBuffers() {
        using namespace OpenMM;
        indices.initialize<int>(cc, 1, "plumedAtomIndices");
        packedPos.initialize(cc, 1, cc.getUseDoublePrecision() ? sizeof(mm_double4) : sizeof(mm_float4), "plumedPositions");
        if (cc.getUseMixedPrecision()) packedCorrection.initialize<mm_float4>(cc, 1, "plumedCorrections");
        packedForce.initialize(cc, 3, cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "plumedForces");
        const std::string code = R"KERNEL(
KERNEL void gatherPlumed(int count, GLOBAL const int* indices,
                        GLOBAL const real4* pos, GLOBAL real4* packed
#ifdef USE_MIXED_PRECISION
                        , GLOBAL const real4* correction, GLOBAL real4* packedCorrection
#endif
                        ) {
    for (int i = GLOBAL_ID; i < count; i += GLOBAL_SIZE) {
        packed[i] = pos[indices[i]];
#ifdef USE_MIXED_PRECISION
        packedCorrection[i] = correction[indices[i]];
#endif
    }
}
KERNEL void scatterPlumed(int count, GLOBAL const int* indices,
                         GLOBAL const real* packed, GLOBAL mm_long* forces) {
    for (int i = GLOBAL_ID; i < count; i += GLOBAL_SIZE) {
        const int atom = indices[i];
        forces[atom] += (mm_long)(packed[3*i]*0x100000000);
        forces[atom+PADDED_NUM_ATOMS] += (mm_long)(packed[3*i+1]*0x100000000);
        forces[atom+2*PADDED_NUM_ATOMS] += (mm_long)(packed[3*i+2]*0x100000000);
    }
}
)KERNEL";
        std::map<std::string, std::string> defines;
        defines["PADDED_NUM_ATOMS"] = std::to_string(cc.getPaddedNumAtoms());
        auto program = cc.compileProgram(code, defines);
        gatherKernel = program->createKernel("gatherPlumed");
        gatherKernel->addArg(0); gatherKernel->addArg(indices);
        gatherKernel->addArg(cc.getPosq()); gatherKernel->addArg(packedPos);
        if (cc.getUseMixedPrecision()) {
            gatherKernel->addArg(cc.getPosqCorrection()); gatherKernel->addArg(packedCorrection);
        }
        scatterKernel = program->createKernel("scatterPlumed");
        scatterKernel->addArg(0); scatterKernel->addArg(indices);
        scatterKernel->addArg(packedForce); scatterKernel->addArg(cc.getLongForceBuffer());
        initialized = true;
    }
    void scatter() {
        int count = requested.size();
        if (count == 0) return;
        if (cc.getUseDoublePrecision()) {
            df.resize(3*count);
            for (int i = 0; i < count; i++) for (int j = 0; j < 3; j++) df[3*i+j] = forces[requested[i]][j];
            packedForce.upload(df);
        } else {
            ff.resize(3*count);
            for (int i = 0; i < count; i++) for (int j = 0; j < 3; j++) ff[3*i+j] = (float)forces[requested[i]][j];
            packedForce.upload(ff);
        }
        scatterKernel->setArg(0, count); scatterKernel->setArg(1, indices);
        scatterKernel->setArg(2, packedForce); scatterKernel->setArg(3, cc.getLongForceBuffer());
        scatterKernel->execute(count);
    }
};
}
#endif
