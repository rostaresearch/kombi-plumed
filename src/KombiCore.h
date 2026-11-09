// Directed hard-bin KOMBI, ported from the corrected Ras/TYK2 implementation.
// Numerical conventions and provenance are documented in README.md.
#ifndef KOMBI_CORE_H
#define KOMBI_CORE_H
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kombi {
using Vec = std::vector<double>;
using Matrix = std::vector<Vec>;
constexpr double infinity = std::numeric_limits<double>::infinity();
inline void require(bool ok, const std::string& why) {
  if (!ok) throw std::runtime_error("KOMBI: " + why);
}
inline Matrix zeros(std::size_t n) { return Matrix(n, Vec(n, 0.0)); }
inline double sum(const Vec& v) { return std::accumulate(v.begin(), v.end(), 0.0); }
inline double logadd(double a, double b) {
  if (a == -infinity) return b;
  if (b == -infinity) return a;
  return std::max(a,b) + std::log1p(std::exp(-std::abs(a-b)));
}
inline void checkMatrix(const Matrix& p) {
  require(!p.empty(), "empty matrix");
  for (const auto& row:p) {
    require(row.size()==p.size(), "matrix must be square");
    for (double x:row) require(std::isfinite(x) && x>=0, "invalid matrix entry");
  }
}
inline Matrix transpose(const Matrix& p) {
  Matrix q=zeros(p.size());
  for (std::size_t i=0;i<p.size();++i) for (std::size_t j=0;j<p.size();++j) q[j][i]=p[i][j];
  return q;
}
inline Matrix submatrix(const Matrix& p,const std::vector<std::size_t>& keep) {
  Matrix q=zeros(keep.size());
  for(std::size_t i=0;i<keep.size();++i) for(std::size_t j=0;j<keep.size();++j) q[i][j]=p[keep[i]][keep[j]];
  return q;
}
inline std::vector<bool> reachable(const Matrix& p,std::size_t start,std::size_t stop=std::numeric_limits<std::size_t>::max()) {
  std::vector<bool> seen(p.size(),false); std::vector<std::size_t> todo{start}; seen[start]=true;
  while(!todo.empty()) {
    auto i=todo.back(); todo.pop_back(); if(i==stop) continue;
    for(std::size_t j=0;j<p.size();++j) if(p[i][j]>0 && !seen[j]) {seen[j]=true;todo.push_back(j);}
  }
  return seen;
}

// Every public matrix here is ROW oriented (origin i -> destination j).
inline double mfpt(Matrix p,std::size_t start,std::size_t target,double dt) {
  checkMatrix(p); require(start<p.size() && target<p.size() && dt>0 && std::isfinite(dt),"invalid MFPT arguments");
  if(start==target) return 0;
  auto forward=reachable(p,start,target), backward=reachable(transpose(p),target);
  std::vector<std::size_t> keep;
  for(std::size_t i=0;i<p.size();++i) if(forward[i]) {
    if(!backward[i]) return infinity; // any reachable closed trap makes unconditional MFPT infinite
    if(i!=target && std::abs(sum(p[i])-1)>1e-10) return infinity;
    keep.push_back(i);
  }
  auto s=std::find(keep.begin(),keep.end(),start)-keep.begin();
  auto t=std::find(keep.begin(),keep.end(),target)-keep.begin();
  p=submatrix(p,keep); const std::size_t n=p.size();
  Vec tau(n,1.0); std::vector<bool> alive(n,true);
  for(std::size_t i=0;i<n;++i) p[i][i]=0;
  for(std::size_t k=0;k<n;++k) {
    if(k==static_cast<std::size_t>(s)||k==static_cast<std::size_t>(t)) continue;
    alive[k]=false; double escape=0;
    for(std::size_t j=0;j<n;++j) if(alive[j]) escape+=p[k][j];
    if(!(escape>0)) return infinity;
    for(std::size_t i=0;i<n;++i) if(alive[i]) {
      const double col=p[i][k];
      for(std::size_t j=0;j<n;++j) if(alive[j]) p[i][j]+=col*(p[k][j]/escape);
      tau[i]+=col*(tau[k]/escape); p[i][k]=0;
    }
  }
  if(!(p[s][t]>0)) return infinity;
  double result=tau[s]/p[s][t]*dt;
  return result>0 && std::isfinite(result)?result:infinity;
}
inline Vec stationaryGth(Matrix p) {
  const std::size_t n=p.size();
  for(std::size_t k=n-1;k>0;--k) {
    double s=0; for(std::size_t j=0;j<k;++j) s+=p[k][j];
    require(s>0,"GTH requires an irreducible communicating class");
    for(std::size_t i=0;i<k;++i) {
      p[i][k]/=s;
      for(std::size_t j=0;j<k;++j) p[i][j]+=p[i][k]*p[k][j];
    }
  }
  Vec r(n,0); r[0]=1;
  for(std::size_t k=1;k<n;++k) for(std::size_t i=0;i<k;++i) r[k]+=r[i]*p[i][k];
  const double total=sum(r); for(auto& x:r) x/=total; return r;
}
inline double meanReturn(const Matrix& p,std::size_t state,double dt) {
  checkMatrix(p); require(state<p.size() && dt>0 && std::isfinite(dt),"invalid return-time arguments");
  auto forward=reachable(p,state), backward=reachable(transpose(p),state);
  std::vector<std::size_t> keep;
  for(std::size_t i=0;i<p.size();++i) if(forward[i]) {
    if(!backward[i] || std::abs(sum(p[i])-1)>1e-10) return infinity;
    keep.push_back(i);
  }
  auto pi=stationaryGth(submatrix(p,keep));
  double prob=pi[std::find(keep.begin(),keep.end(),state)-keep.begin()];
  return prob>0?dt/prob:infinity;
}

// Parameters: [a_1..a_G, b_1..b_G, sigma_1..sigma_G].
inline std::pair<double,double> gaussian(double x,const Vec& p) {
  require(p.size()>0 && p.size()%3==0 && std::isfinite(x),"invalid Gaussian input");
  const auto g=p.size()/3; double energy=0,derivative=0;
  for(std::size_t k=0;k<g;++k) {
    require(std::isfinite(p[k]) && p[k]>=0 && std::isfinite(p[g+k]) && std::isfinite(p[2*g+k]) && p[2*g+k]>0,"invalid Gaussian parameters");
    const double d=x-p[g+k],sigma=p[2*g+k],v=p[k]*std::exp(-0.5*(d/sigma)*(d/sigma));
    energy+=v; derivative-=v*d/(sigma*sigma);
  }
  require(std::isfinite(energy)&&std::isfinite(derivative),"nonfinite bias or derivative");
  return {energy,derivative};
}
inline Vec biasOnGrid(const Vec& q,const Vec& p) {
  Vec u; for(double x:q) u.push_back(gaussian(x,p).first); return u;
}
inline std::array<double,3> gaussianND(double x,double y,const Vec& p,std::size_t dim) {
  if(dim==1){auto v=gaussian(x,p);return {{v.first,v.second,0}};}
  require(dim==2 && p.size()>0 && p.size()%5==0 && std::isfinite(x) && std::isfinite(y),"invalid 2D Gaussian input");
  const std::size_t g=p.size()/5;std::array<double,3> out{{0,0,0}};
  for(std::size_t k=0;k<g;++k) {
    for(std::size_t b=0;b<5;++b)require(std::isfinite(p[b*g+k]),"nonfinite 2D Gaussian parameter");
    require(p[k]>=0 && p[3*g+k]>0 && p[4*g+k]>0,"invalid Gaussian amplitude/width");
    const double dx=x-p[g+k],dy=y-p[2*g+k],sx=p[3*g+k],sy=p[4*g+k];
    const double v=p[k]*std::exp(-.5*((dx/sx)*(dx/sx)+(dy/sy)*(dy/sy)));
    out[0]+=v;out[1]-=v*dx/(sx*sx);out[2]-=v*dy/(sy*sy);
  }
  for(double v:out)require(std::isfinite(v),"nonfinite 2D bias/force");
  return out;
}
inline Vec biasOnPoints(const Vec& x,const Vec& y,const Vec& p,std::size_t dim) {
  if(dim==1)return biasOnGrid(x,p);
  require(x.size()==y.size(),"point coordinate mismatch");Vec u(x.size());
  for(std::size_t i=0;i<x.size();++i)u[i]=gaussianND(x[i],y[i],p,dim)[0];
  return u;
}
inline std::size_t digitize(double x,const Vec& q) {
  // Deliberately numpy.digitize(x,q), without subtracting one: Ras convention.
  require(std::isfinite(x) && x>=q.front() && x<q.back(),"CV outside fixed grid [GRID_MIN,GRID_MAX)");
  return std::upper_bound(q.begin(),q.end(),x)-q.begin();
}
inline Matrix counts(const std::vector<int>& bins,std::size_t n,std::size_t lag) {
  require(lag>0,"LAG must be positive"); Matrix c=zeros(n);
  for(std::size_t k=lag;k<bins.size();++k) if(bins[k-lag]>=0 && bins[k]>=0 && bins[k-lag]<static_cast<int>(n) && bins[k]<static_cast<int>(n)) c[bins[k-lag]][bins[k]]+=1;
  return c;
}
struct Accumulator {
  Matrix count,logDen;
  explicit Accumulator(std::size_t n):count(zeros(n)),logDen(n,Vec(n,-infinity)){}
  void add(const Matrix& c,const Vec& u,double kbt) {
    checkMatrix(c); require(c.size()==count.size() && u.size()==c.size() && kbt>0 && std::isfinite(kbt),"invalid DHAM input");
    for(double x:u) require(std::isfinite(x),"nonfinite window bias");
    for(std::size_t i=0;i<c.size();++i) {
      const double exposure=sum(c[i]);
      for(std::size_t j=0;j<c.size();++j) {
        count[i][j]+=c[i][j]; require(std::isfinite(count[i][j]) && count[i][j]<9007199254740992.0,"count overflow");
        if(exposure>0) {
          const double term=std::log(exposure)+(u[i]-u[j])/(2*kbt);
          require(std::isfinite(term),"DHAM exponential outside numerical range");
          logDen[i][j]=logadd(logDen[i][j],term);
        }
      }
    }
  }
  Matrix model() const {
    Matrix p=zeros(count.size());
    for(std::size_t i=0;i<count.size();++i) {
      Vec l(count.size(),-infinity); double normal=-infinity;
      for(std::size_t j=0;j<count.size();++j) if(count[i][j]>0) {
        require(std::isfinite(logDen[i][j]),"missing exposure for observed edge");
        l[j]=std::log(count[i][j])-logDen[i][j]; normal=logadd(normal,l[j]);
      }
      for(std::size_t j=0;j<count.size();++j) if(count[i][j]>0) {
        p[i][j]=std::exp(l[j]-normal);
        require(p[i][j]>0 && std::isfinite(p[i][j]),"observed edge underflow; refusing to erase support");
      }
    }
    return p;
  }
};
inline Matrix tilt(const Matrix& p,const Vec& u,double kbt) {
  checkMatrix(p); require(u.size()==p.size() && std::isfinite(kbt) && kbt>0,"invalid tilt input");
  Matrix b=zeros(p.size());
  for(std::size_t i=0;i<p.size();++i) {
    Vec l(p.size(),-infinity); double normal=-infinity;
    for(std::size_t j=0;j<p.size();++j) if(p[i][j]>0) {
      require(std::isfinite(u[i]) && std::isfinite(u[j]),"nonfinite tilt bias");
      l[j]=std::log(p[i][j])+(u[i]-u[j])/(2*kbt); normal=logadd(normal,l[j]);
    }
    for(std::size_t j=0;j<p.size();++j) if(p[i][j]>0) {
      b[i][j]=std::exp(l[j]-normal);
      require(std::isfinite(b[i][j]) && b[i][j]>0,"tilted edge underflow; refusing to erase support");
    }
  }
  return b;
}

struct Settings {
  double low=0,high=40,target=40,kbt=0.5961612777,tiltKbt=0.5981;
  double lowY=0,highY=40,targetY=40;
  double ampMin=.1,ampMax=5,widthMin=.7,widthMax=2,randomAmp=.6,randomWidthMin=.7,randomWidthMax=1,tol=1e-6;
  std::size_t bins=100,hills=10,lag=1,pace=10000,stride=1,randomTrials=1000,maxIter=15000;
  std::size_t dim=1,binsY=1;
  int gridMode=0; // 0: RAS; 1: ordinary centers; 2/3: corrected legacy 2D/1D conventions
  bool clipGrid=false;
  bool stopAtTarget=false;
  double targetRadius=.1;
  std::uint64_t seed=1;
  std::size_t states()const{return bins*binsY;}
  std::size_t parameters()const{return (1+2*dim)*hills;}
  bool hitsTarget(double x,double y=0)const {
    if(!stopAtTarget)return false;
    if(gridMode==0)return x>=target;
    const double dx=x-target,dy=dim==2?y-targetY:0;
    return dx*dx+dy*dy<=targetRadius*targetRadius;
  }
  Vec axis(double lo,double hi,std::size_t n)const {
    const double spacing=(hi-lo)/((gridMode==1||gridMode==3)?n:n-1);
    Vec q(n);
    for(std::size_t i=0;i<n;++i) {
      q[i]=lo+spacing*double(i);
      if(gridMode==1)q[i]+=spacing/2;
      if(gridMode==2||gridMode==3)q[i]-=spacing/2;
    }
    if(gridMode==0)q.back()=hi;
    if(gridMode==2)q.back()=hi-spacing/2;
    return q;
  }
  Vec grid() const {auto ax=axis(low,high,bins);Vec q(states());for(std::size_t i=0;i<q.size();++i)q[i]=ax[i%bins];return q;}
  Vec ygrid()const {Vec q(states(),0);if(dim==2){auto ay=axis(lowY,highY,binsY);for(std::size_t i=0;i<q.size();++i)q[i]=ay[i/bins];}return q;}
  std::size_t state(double x,double y=0)const {
    auto index=[&](double v,double lo,double hi,std::size_t n) {
      require(std::isfinite(v),"nonfinite CV");
      if(gridMode==3) {
        // Historical 1D hard-bin driver: digitize against n+1 edges, without -1.
        // Index n and above is invalid; count only pairs whose two endpoints are valid.
        const double spacing=(hi-lo)/n;
        Vec edges(n+1);for(std::size_t i=0;i<=n;++i)edges[i]=lo+spacing*double(i);edges.back()=hi;
        return std::min(n,std::size_t(std::upper_bound(edges.begin(),edges.end(),v)-edges.begin()));
      }
      if(!clipGrid)require(v>=lo && v<hi,"CV outside fixed grid [GRID_MIN,GRID_MAX)");
      if(gridMode==1) {
        if(v<=lo)return std::size_t(0);
        if(v>=hi)return n-1;
        const double spacing=(hi-lo)/n;
        Vec edges(n+1);for(std::size_t i=0;i<=n;++i)edges[i]=lo+spacing*double(i);edges.back()=hi;
        return std::size_t(std::upper_bound(edges.begin(),edges.end(),v)-edges.begin()-1);
      }
      const double spacing=(hi-lo)/(n-1);
      Vec edges(n);for(std::size_t i=0;i<n;++i)edges[i]=lo+spacing*double(i);edges.back()=hi;
      return std::min(n-1,std::size_t(std::upper_bound(edges.begin(),edges.end(),v)-edges.begin()));
    };
    return index(x,low,high,bins)+(dim==2?bins*index(y,lowY,highY,binsY):0);
  }
  void validate() const {
    for(double x:{low,high,target,lowY,highY,targetY,kbt,tiltKbt,ampMin,ampMax,widthMin,widthMax,randomAmp,randomWidthMin,randomWidthMax,tol,targetRadius}) require(std::isfinite(x),"nonfinite setting");
    require(targetRadius>=0,"TARGET_RADIUS must be nonnegative");
    require(bins>=3 && bins<=1000 && hills>0 && hills<=1000,"BINS must be 3..1000; NGAUSSIANS must be 1..1000");
    require((dim==1 && binsY==1)||(dim==2 && binsY>=3 && binsY<=1000),"invalid dimension or second grid size");
    require(states()<=4096,"at most 4096 total grid states are supported");
    require(gridMode>=0 && gridMode<=3 && (gridMode!=0||dim==1) && (gridMode!=2||dim==2) && (gridMode!=3||dim==1),"invalid GRID_MODE for number of CVs");
    require(gridMode!=0||!clipGrid,"RAS convention uses strict grid bounds");
    require(gridMode!=3||!clipGrid,"LEGACY_1D skips invalid pairs; CLIP_GRID is incompatible");
    require(high>low && target>=low && target<=high,"invalid grid/target");
    require(dim==1||(highY>lowY && targetY>=lowY && targetY<=highY),"invalid second grid/target");
    require(kbt>0 && tiltKbt>0 && tol>0,"KBT, TILT_KBT and TOL must be positive");
    require(ampMin>=0 && ampMax>=ampMin && widthMin>0 && widthMax>=widthMin,"invalid parameter bounds");
    require(randomAmp>=ampMin && randomAmp<=ampMax && randomWidthMin>=widthMin && randomWidthMax<=widthMax && randomWidthMax>=randomWidthMin,"random starts outside bounds");
    require(stride>0 && lag>0 && pace>0 && pace%stride==0 && pace/stride>lag && randomTrials>0 && maxIter>0,"invalid STRIDE/LAG/PACE/search budget");
  }
  std::string signature() const {
    std::ostringstream o; o<<std::setprecision(17)<<low<<' '<<high<<' '<<target<<' '<<kbt<<' '<<tiltKbt<<' '<<ampMin<<' '<<ampMax<<' '<<widthMin<<' '<<widthMax<<' '<<randomAmp<<' '<<randomWidthMin<<' '<<randomWidthMax<<' '<<tol<<' '<<bins<<' '<<hills<<' '<<lag<<' '<<pace<<' '<<stride<<' '<<randomTrials<<' '<<maxIter<<' '<<seed;
    o<<' '<<dim<<' '<<binsY<<' '<<lowY<<' '<<highY<<' '<<targetY<<' '<<gridMode<<' '<<clipGrid<<' '<<stopAtTarget<<' '<<targetRadius;return o.str();
  }
};
struct OptResult {Vec params;double bestRandom=infinity,selected=infinity;int iterations=0,status=0;};
inline Vec validatedGrid(const Settings& c) {c.validate();return c.grid();}
// L-BFGS-B 3.0, with finite differences and finite-best fallback.
OptResult optimize(const Settings&,const Matrix&,const Vec&,std::size_t,std::size_t,std::size_t,double,const Vec& y=Vec{});

struct Decision {
  Matrix windowCounts,model; Vec applied,next,bias; std::vector<std::size_t> keep;
  std::size_t start=0,target=0,window=0,skipped=0; double randomMfpt=-1,selectedMfpt=-1; int iterations=0,status=0;
};

inline void writeVec(std::ostream& o,const Vec& v) {for(double x:v) o<<x<<' ';o<<'\n';}
inline void writeMatrix(std::ostream& o,const Matrix& p) {for(const auto& row:p)writeVec(o,row);}
inline double readDouble(std::istream& in,bool allowNegInf=false) {
  std::string token; require(bool(in>>token),"truncated state");std::size_t used=0;double x;
  try{x=std::stod(token,&used);}catch(...){throw std::runtime_error("KOMBI: malformed state number");}
  require(used==token.size() && (std::isfinite(x)||(allowNegInf && x==-infinity)),"invalid state number"); return x;
}
inline Vec readVec(std::istream& in,std::size_t n,bool allowNegInf=false) {Vec v(n);for(auto& x:v)x=readDouble(in,allowNegInf);return v;}
inline void readMatrix(std::istream& in,Matrix& p,bool allowNegInf=false) {for(auto& row:p)row=readVec(in,row.size(),allowNegInf);}
inline std::uint64_t checksum(const std::string& text,std::uint64_t h=UINT64_C(14695981039346656037)) {
  for(unsigned char c:text){h^=c;h*=UINT64_C(1099511628211);}return h;
}

class Engine {
public:
  Settings cfg; Vec grid,gridY,params,replayParams; Accumulator acc; Matrix current; Vec visits;
  std::deque<std::size_t> recent; std::size_t samples=0,windows=0,skipped=0;
  long long lastStep=-1; double dt=0,lastCv=0,lastY=0,lastMfpt=-1,replayMfpt=-1; std::size_t replayWindows=0;
  long long firstPassageStep=-1;
  std::uint64_t historyDigest=UINT64_C(14695981039346656037);
  explicit Engine(const Settings& c,const Vec& initial):cfg(c),grid(validatedGrid(c)),gridY(c.ygrid()),params(initial),replayParams(initial),acc(c.states()),current(zeros(c.states())),visits(c.states()+1,0) {
    cfg.validate(); require(params.size()==cfg.parameters(),"INITIAL_PARAMS must have (1+2*dimensions)*NGAUSSIANS entries");gaussianND(cfg.low,cfg.lowY,params,cfg.dim);
    for(std::size_t i=0;i<cfg.hills;++i)require(params[i]<=cfg.ampMax,"initial Gaussian exceeds AMP_MAX");
  }
  const Vec& appliedAt(long long step) const {return step==lastStep?replayParams:params;}
  // Called once by PLUMED update(), after force evaluation. Step zero is not a sample.
  bool observe(long long step,double x,double timestep,Decision& dec,double y=0) {
    require(step>=0 && std::isfinite(timestep) && timestep>0,"invalid MD step/timestep");
    require(std::isfinite(x) && std::isfinite(y),"nonfinite CV");
    const bool hit=cfg.hitsTarget(x,y);
    if(!hit)cfg.state(x,y);
    if(dt==0)dt=timestep;
    require(dt==timestep,"timestep changed across restart");
    if(step==lastStep) {
      require(std::abs(x-lastCv)<=1e-10*std::max(1.0,std::abs(x)) && std::abs(y-lastY)<=1e-10*std::max(1.0,std::abs(y)),"replayed checkpoint step has different CV"); return false;
    }
    require((lastStep<0 && step==0)||step==lastStep+1,"nonconsecutive steps or mismatched checkpoint; every MD step must be supplied");
    require(firstPassageStep<0,"run already reached its stopping target");
    replayParams=params;replayMfpt=lastMfpt;replayWindows=windows;lastStep=step;lastCv=x;lastY=y;
    // A terminal frame can lie beyond the fixed Ras grid. Stop before counting it,
    // without reconstructing an incomplete window or inventing an out-of-grid bin.
    if(hit){firstPassageStep=step;return false;}
    if(step==0 || step%static_cast<long long>(cfg.stride)!=0) return false;
    const auto state=cfg.state(x,y);recent.push_back(state);visits[state]+=1;
    if(recent.size()>cfg.lag) {if(recent.front()<cfg.states() && state<cfg.states())current[recent.front()][state]+=1;else ++skipped;recent.pop_front();}
    ++samples;
    if(samples<cfg.pace/cfg.stride)return false;
    dec.window=windows;dec.skipped=skipped;dec.windowCounts=current;dec.applied=params;dec.bias=biasOnPoints(grid,gridY,params,cfg.dim);
    // The Ras neighbor-only tilt is equivalent to the all-edge tilt ONLY here.
    // Fail instead of silently discarding observed non-neighbor evidence.
    if(cfg.gridMode==0)for(std::size_t i=0;i<cfg.states();++i)for(std::size_t j=0;j<cfg.states();++j)
      require((i<=j?j-i:i-j)<=1 || current[i][j]==0,"non-neighbor transition; decrease sampling stride or use coarser grid");
    acc.add(current,dec.bias,cfg.kbt); Matrix p=acc.model(); auto pt=transpose(p);
    for(std::size_t i=0;i<cfg.states();++i) if(sum(p[i])>0 && sum(pt[i])>0)dec.keep.push_back(i);
    require(!dec.keep.empty(),"no observed working chain");
    std::vector<bool> retained(cfg.states(),false);for(auto i:dec.keep)retained[i]=true;
    for(auto i:dec.keep)for(std::size_t j=0;j<cfg.states();++j)
      require(retained[j] || p[i][j]==0,"trimming would discard probability; preserve evidence and extend sampling");
    dec.model=submatrix(p,dec.keep);
    for(const auto& row:dec.model)require(std::abs(sum(row)-1)<1e-10,"trimming would discard probability; preserve evidence and extend sampling");
    Settings targetCfg=cfg;targetCfg.clipGrid=true;
    const std::size_t final=cfg.gridMode==0?std::upper_bound(grid.begin(),grid.end(),cfg.target)-grid.begin()-1:std::min(cfg.states()-1,targetCfg.state(cfg.target,cfg.targetY));
    auto nearest=[&](std::size_t st) {
      std::size_t k=0;double best=infinity;
      for(std::size_t i=0;i<dec.keep.size();++i) {
        const auto si=dec.keep[i];
        // Historical drivers select nearest in integer bin space (first on ties).
        const double dx=cfg.gridMode==1?grid[si]-grid[st]:double(si%cfg.bins)-double(st%cfg.bins);
        const double dy=cfg.gridMode==1?gridY[si]-gridY[st]:double(si/cfg.bins)-double(st/cfg.bins);
        const double d=dx*dx+dy*dy;if(d<best){best=d;k=i;}
      }
      return k;
    };
    // The corrected 2D driver starts the objective at the most visited bin;
    // Ras and 1D use the last sampled bin. Ties select the smallest bin index.
    const auto startState=cfg.gridMode==2?std::size_t(std::max_element(visits.begin(),visits.end()-1)-visits.begin()):std::min(state,cfg.states()-1);
    dec.start=nearest(startState);dec.target=nearest(final);
    if(cfg.gridMode==2) {
      dec.start=0;double best=infinity;
      for(std::size_t i=0;i<dec.keep.size();++i) {
        const double d=std::abs(double(dec.keep[i])-double(startState));
        if(d<best){best=d;dec.start=i;}
      }
    }
    Vec q,qy;for(auto i:dec.keep){q.push_back(grid[i]);qy.push_back(gridY[i]);}
    auto result=optimize(cfg,dec.model,q,dec.start,dec.target,windows,dt*cfg.stride*cfg.lag,qy);
    dec.randomMfpt=result.bestRandom; dec.selectedMfpt=result.selected;dec.iterations=result.iterations;dec.status=result.status;
    params=result.params;dec.next=params;lastMfpt=result.selected;
    ++windows; current=zeros(cfg.states());std::fill(visits.begin(),visits.end(),0);recent.clear();samples=0;skipped=0;return true;
  }
  std::string save() const {
    std::ostringstream o;o<<std::setprecision(17)<<"KOMBI_STATE_V1\n"<<cfg.signature()<<'\n';
    o<<lastStep<<' '<<dt<<' '<<lastCv<<' '<<lastY<<' '<<lastMfpt<<' '<<samples<<' '<<windows<<' '<<replayMfpt<<' '<<replayWindows<<' '<<historyDigest<<' '<<skipped<<' '<<firstPassageStep<<'\n';
    writeVec(o,params);writeVec(o,replayParams);writeMatrix(o,acc.count);writeMatrix(o,acc.logDen);writeMatrix(o,current);writeVec(o,visits);
    o<<recent.size()<<' ';for(auto i:recent)o<<i<<' ';o<<"\nEND_KOMBI_STATE\n";
    const std::string payload=o.str();o<<"CHECKSUM "<<checksum(payload)<<'\n';return o.str();
  }
  void restore(const std::string& text) {
    const auto at=text.rfind("CHECKSUM ");require(at!=std::string::npos,"missing checkpoint checksum");
    const std::string payload=text.substr(0,at);std::istringstream check(text.substr(at+9));std::uint64_t expected;std::string extra;
    require(bool(check>>expected) && !(check>>extra) && expected==checksum(payload),"checkpoint checksum mismatch");
    std::istringstream in(payload);std::string line;std::getline(in,line);require(line=="KOMBI_STATE_V1","unsupported state version");
    std::getline(in,line);require(line==cfg.signature(),"restart settings differ from checkpoint");
    require(bool(in>>lastStep),"truncated checkpoint");dt=readDouble(in);lastCv=readDouble(in);lastY=readDouble(in);lastMfpt=readDouble(in);
    require(bool(in>>samples>>windows) && lastStep>=0 && dt>0 && samples<cfg.pace/cfg.stride,"invalid checkpoint metadata");
    replayMfpt=readDouble(in);require(bool(in>>replayWindows>>historyDigest>>skipped>>firstPassageStep) && replayWindows<=windows,"invalid replay metadata");
    require(firstPassageStep==-1||(firstPassageStep==lastStep && cfg.hitsTarget(lastCv,lastY)),"invalid first-passage metadata");
    params=readVec(in,cfg.parameters());replayParams=readVec(in,cfg.parameters());gaussianND(lastCv,lastY,params,cfg.dim);gaussianND(lastCv,lastY,replayParams,cfg.dim);
    if(firstPassageStep<0)cfg.state(lastCv,lastY);
    readMatrix(in,acc.count);readMatrix(in,acc.logDen,true);readMatrix(in,current);checkMatrix(acc.count);checkMatrix(current);
    visits=readVec(in,cfg.states()+1);
    for(double v:visits)require(v>=0 && v==std::floor(v),"invalid visit count");
    require(sum(visits)==double(samples),"checkpoint visit count mismatch");
    std::size_t n;require(bool(in>>n) && n<=cfg.lag && n==std::min(samples,cfg.lag),"invalid lag history");recent.clear();
    for(std::size_t k=0;k<n;++k){std::size_t i;require(bool(in>>i)&&(i<cfg.states()||(cfg.gridMode==3&&i==cfg.states())),"invalid state index");recent.push_back(i);}
    require(bool(in>>line)&&line=="END_KOMBI_STATE","missing state terminator");require(!(in>>line),"trailing state data");
    double total=0;for(const auto& r:current)total+=sum(r);
    require(total+skipped==double(samples>cfg.lag?samples-cfg.lag:0),"checkpoint count/history mismatch");
    acc.model(); // verifies observed support and exposure consistency
  }
};
} // namespace kombi
#endif
