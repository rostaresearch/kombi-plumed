// Small text interface for independent Python/reference tests of compiled C++.
#include "../src/KombiCore.h"
#include <iostream>
using namespace kombi;
int main(int argc,char** argv) {
  try {
    require(argc==2,"one probe mode required");std::string mode=argv[1];std::cout<<std::setprecision(17);
    std::size_t n;std::cin>>n;
    if(mode=="grid") {
      Settings c;c.bins=n;std::size_t m;std::cin>>c.dim>>c.binsY>>c.gridMode>>c.clipGrid>>c.low>>c.high>>c.lowY>>c.highY>>m;c.target=c.high;c.targetY=c.highY;c.validate();
      writeVec(std::cout,c.grid());writeVec(std::cout,c.ygrid());
      for(std::size_t i=0;i<m;++i){double x,y;std::cin>>x>>y;std::cout<<c.state(x,y)<<' ';}std::cout<<'\n';
    } else if(mode=="bias2") {
      std::size_t m;std::cin>>m;auto p=readVec(std::cin,5*n);
      for(std::size_t i=0;i<m;++i){double x,y;std::cin>>x>>y;auto v=gaussianND(x,y,p,2);std::cout<<v[0]<<' '<<v[1]<<' '<<v[2]<<'\n';}
    } else if(mode=="counts") {
      std::size_t lag,m;std::cin>>lag>>m;std::vector<int> b(m);for(auto& x:b)std::cin>>x;writeMatrix(std::cout,counts(b,n,lag));
    } else if(mode=="dham") {
      std::size_t w;double kbt;std::cin>>w>>kbt;Accumulator acc(n);
      for(std::size_t k=0;k<w;++k){Matrix c=zeros(n);readMatrix(std::cin,c);auto u=readVec(std::cin,n);acc.add(c,u,kbt);}writeMatrix(std::cout,acc.model());
    } else if(mode=="tilt") {
      double kbt;std::cin>>kbt;auto u=readVec(std::cin,n);Matrix p=zeros(n);readMatrix(std::cin,p);writeMatrix(std::cout,tilt(p,u,kbt));
    } else if(mode=="mfpt"||mode=="return"||mode=="stationary") {
      std::size_t start,target;double dt;std::cin>>start>>target>>dt;Matrix p=zeros(n);readMatrix(std::cin,p);
      if(mode=="mfpt")std::cout<<mfpt(p,start,target,dt)<<'\n';
      if(mode=="return")std::cout<<meanReturn(p,start,dt)<<'\n';
      if(mode=="stationary")writeVec(std::cout,stationaryGth(p));
    } else if(mode=="bias") {
      std::size_t m;std::cin>>m;auto p=readVec(std::cin,3*n);auto x=readVec(std::cin,m);
      for(double v:x){auto out=gaussian(v,p);std::cout<<out.first<<' '<<out.second<<'\n';}
    } else if(mode=="optimize") {
      Settings c;std::size_t s,t;std::cin>>s>>t>>c.seed;c.bins=n;c.low=0;c.high=n-1;c.target=c.high;
      c.hills=2;c.randomTrials=40;c.maxIter=200;c.kbt=c.tiltKbt=.5981;
      Matrix p=zeros(n);readMatrix(std::cin,p);auto out=optimize(c,p,c.grid(),s,t,0,.002);
      std::cout<<out.bestRandom<<' '<<out.selected<<' '<<out.status<<'\n';writeVec(std::cout,out.params);
    } else require(false,"unknown probe mode");
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
}
