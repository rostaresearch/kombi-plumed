#include "KombiCore.h"
#include <mutex>
extern "C" {
#include "../third_party/lbfgsb/lbfgsb.h"
}

namespace kombi {
namespace {
// Explicit SplitMix64 conversion: reproducible across standard libraries.
// NumPy uses a different generator; identical seeds are not identical proposals.
class Random {
  std::uint64_t state;
public:
  explicit Random(std::uint64_t seed):state(seed){}
  double uniform(double lo,double hi) {
    std::uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    z^=z>>31;
    return lo+(hi-lo)*double(z>>11)*0x1.0p-53;
  }
};
}
OptResult optimize(const Settings& cfg,const Matrix& p,const Vec& q,std::size_t start,std::size_t target,std::size_t window,double dt,const Vec& qy) {
  cfg.validate(); require(p.size()==q.size(),"working grid/model mismatch");
  auto objective=[&](const Vec& params,double step) {
    try {
      auto b=tilt(p,biasOnPoints(q,qy,params,cfg.dim),cfg.tiltKbt);
      double f=start==target?meanReturn(b,start,step):mfpt(b,start,target,step);
      return f>0 && std::isfinite(f)?f:infinity;
    } catch(const std::runtime_error&) {return infinity;}
  };
  OptResult result;
  Random rng(cfg.seed^(UINT64_C(0xd1b54a32d192ed03)*(window+1)));
  for(std::size_t trial=0;trial<cfg.randomTrials;++trial) {
    Vec x(cfg.parameters(),cfg.randomAmp);
    for(std::size_t d=0;d<cfg.dim;++d)for(std::size_t k=0;k<cfg.hills;++k)x[(1+d)*cfg.hills+k]=rng.uniform(d?cfg.lowY:cfg.low,d?cfg.highY:cfg.high);
    for(std::size_t d=0;d<cfg.dim;++d)for(std::size_t k=0;k<cfg.hills;++k)x[(1+cfg.dim+d)*cfg.hills+k]=rng.uniform(cfg.randomWidthMin,cfg.randomWidthMax);
    double f=objective(x,dt);
    if(f<result.bestRandom) {result.params=x;result.bestRandom=f;}
  }
  require(std::isfinite(result.bestRandom),"no finite MFPT candidate on directed graph; no random bias was applied");
  result.selected=result.bestRandom;
  Vec x=result.params,lo(cfg.parameters()),hi(cfg.parameters()),gradient(cfg.parameters());
  for(std::size_t k=0;k<cfg.hills;++k) {
    lo[k]=cfg.ampMin;hi[k]=cfg.ampMax;
    for(std::size_t d=0;d<cfg.dim;++d) {
      lo[(1+d)*cfg.hills+k]=d?cfg.lowY:cfg.low;hi[(1+d)*cfg.hills+k]=d?cfg.highY:cfg.high;
      lo[(1+cfg.dim+d)*cfg.hills+k]=cfg.widthMin;hi[(1+cfg.dim+d)*cfg.hills+k]=cfg.widthMax;
    }
  }
  integer n=x.size(),m=10,task=START,iprint=-1,csave=0;
  std::vector<integer> nbd(n,2),iwa(3*n,0);
  Vec wa((2*m+5)*n+11*m*m+8*m,0);
  integer isave[44]={0};logical lsave[4]={0};double dsave[29]={0};
  double f=0,factr=cfg.tol/std::numeric_limits<double>::epsilon(),pgtol=cfg.tol;
  // The vendored Fortran-to-C solver has process-global static work variables.
  // Serialize the entire reverse-communication solve across PLUMED instances.
  static std::mutex solverMutex;std::lock_guard<std::mutex> lock(solverMutex);
  std::size_t evaluations=0;
  const double localStep=cfg.gridMode==0?.1:1.0;
  while(true) {
    setulb(&n,&m,x.data(),lo.data(),hi.data(),nbd.data(),&f,gradient.data(),&factr,&pgtol,wa.data(),iwa.data(),&task,&iprint,&csave,lsave,isave,dsave);
    if(IS_FG(task)) {
      // Original SciPy helper uses time_step=0.1 for local optimization.
      f=objective(x,localStep);if(!std::isfinite(f)){task=ABNORMAL;break;}
      double physical=objective(x,dt);
      if(physical<result.selected){result.selected=physical;result.params=x;}
      bool valid=true;
      for(integer k=0;k<n;++k) {
        if(hi[k]==lo[k]) {gradient[k]=0;continue;}
        double h=1e-8;
        if(x[k]+h>hi[k])h=-h;
        if(x[k]+h<lo[k])h=hi[k]-x[k];
        Vec shifted=x;shifted[k]+=h;h=shifted[k]-x[k];
        if(h==0){valid=false;break;}
        double fh=objective(shifted,localStep);gradient[k]=(fh-f)/h;
        if(!std::isfinite(gradient[k])){valid=false;break;}
      }
      if(!valid || ++evaluations>cfg.maxIter*25){task=ABNORMAL;break;}
    } else if(task==NEW_X) {
      ++result.iterations;
      if(static_cast<std::size_t>(result.iterations)>=cfg.maxIter){task=STOP_ITER;break;}
    } else break;
  }
  result.status=static_cast<int>(task);
  // Stronger finite-best safeguard: never replace the random candidate by a worse local iterate.
  require(std::isfinite(result.selected)&&result.selected>0,"invalid selected objective");
  for(integer k=0;k<n;++k)require(std::isfinite(result.params[k])&&result.params[k]>=lo[k]&&result.params[k]<=hi[k],"optimizer selected parameters outside bounds");
  return result;
}
}
