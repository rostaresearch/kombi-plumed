#include "bias/Bias.h"
#include "core/ActionRegister.h"
#include "core/PlumedMain.h"
#include "tools/OFile.h"
#include "tools/IFile.h"
#include "KombiCore.h"
#include <memory>

namespace PLMD { namespace bias {
//+PLUMEDOC BIAS KOMBI
/*
Directed hard-bin kinetics-optimized biasing on one or two nonperiodic scalar CVs.
The corrected Ras/TYK2 DHAM estimator uses separate constant-bias windows;
positive Gaussian parameters minimize the model first-passage (or return) time.
Use the README shipped with this extension for conventions, tests and examples.
KBT and TILT_KBT are energies in the active PLUMED UNITS. TARGET selects the
model objective; STOP_AT_TARGET additionally enables physical first-hit stopping.
The model MFPT is not an unbiased off-rate.
*/
//+ENDPLUMEDOC
class Kombi : public Bias {
  std::unique_ptr<kombi::Engine> engine;
  OFile state,history;
  std::string stateName,historyName;
  unsigned checkpointStride;
  Value *mfptValue,*windowValue,*fptValue;
  void checkpoint() {state.rewind();state.printf("%s",engine->save().c_str());state.flush();}
  void record(const kombi::Decision& d) {
    std::ostringstream o;o<<std::setprecision(17);
    o<<"WINDOW "<<d.window<<" STEP "<<getStep()<<" SAMPLES "<<engine->cfg.pace/engine->cfg.stride<<" LAG "<<engine->cfg.lag<<" SKIPPED_PAIRS "<<d.skipped<<'\n';
    o<<"APPLIED\n";kombi::writeVec(o,d.applied);
    o<<"BIAS_ON_GRID\n";kombi::writeVec(o,d.bias);
    o<<"COUNTS\n";kombi::writeMatrix(o,d.windowCounts);
    o<<"WORKING "<<d.keep.size()<<" START "<<d.start<<" TARGET "<<d.target<<'\n';for(auto i:d.keep)o<<i<<' ';o<<'\n';
    o<<"MODEL_ROW_ORIENTED\n";kombi::writeMatrix(o,d.model);
    o<<"MFPT "<<d.randomMfpt<<' '<<d.selectedMfpt<<" ITERATIONS "<<d.iterations<<" STATUS "<<d.status<<'\n';
    o<<"NEXT\n";kombi::writeVec(o,d.next);o<<"END_WINDOW\n";
    const std::string entry=o.str();history.printf("%s",entry.c_str());history.flush();
    engine->historyDigest=kombi::checksum(entry,engine->historyDigest);
  }
public:
  explicit Kombi(const ActionOptions&);
  static void registerKeywords(Keywords&);
  void calculate() override;
  void update() override;
};
PLUMED_REGISTER_ACTION(Kombi,"KOMBI")

void Kombi::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);if(keys.reserved("ARG"))keys.use("ARG");
  keys.add("compulsory","GRID_MIN","lower fixed-grid endpoint in CV units");
  keys.add("compulsory","GRID_MAX","upper fixed-grid endpoint (exclusive for sampled CVs)");
  keys.add("compulsory","BINS","100","number of qspace points, using the corrected Ras digitization");
  keys.add("optional","GRID_MODE","RAS for phosphate unbinding; CENTERS for ordinary bin centers; LEGACY_1D and LEGACY_2D reproduce corrected October model-potential conventions");
  keys.addFlag("CLIP_GRID",false,"map finite out-of-domain CVs to edge bins, as in legacy model-potential drivers; unavailable in RAS mode");
  keys.add("compulsory","TARGET","model target in CV units, may equal GRID_MAX");
  keys.addFlag("STOP_AT_TARGET",false,"request MD stop at first target hit: x>=TARGET in RAS mode; within TARGET_RADIUS otherwise");
  keys.add("compulsory","TARGET_RADIUS","0.1","physical stopping radius in CV units for non-RAS modes; used only with STOP_AT_TARGET");
  keys.add("compulsory","KBT","thermal energy for DHAM reconstruction in active energy units");
  keys.add("optional","TILT_KBT","thermal energy used for optimizer tilt; defaults to KBT; historical Ras value is 0.5981 kcal/mol");
  keys.add("compulsory","PACE","10000","MD steps per constant-bias window; multiple of SAMPLE_STRIDE");
  keys.add("compulsory","SAMPLE_STRIDE","1","MD steps between CV samples; forces still computed every step");
  keys.add("compulsory","LAG","1","lag in sampled frames, within each bias window");
  keys.add("compulsory","NGAUSSIANS","10","number of positive Gaussian hills in each replacement bias");
  keys.add("compulsory","AMP_MIN","minimum optimized amplitude per hill");
  keys.add("compulsory","AMP_MAX","maximum amplitude per hill");
  keys.add("compulsory","SIGMA_MIN","minimum Gaussian standard deviation in CV units");
  keys.add("compulsory","SIGMA_MAX","maximum Gaussian standard deviation in CV units");
  keys.add("compulsory","RANDOM_AMP","fixed amplitude for random proposals");
  keys.add("compulsory","RANDOM_SIGMA_MIN","lower width for random proposals");
  keys.add("compulsory","RANDOM_SIGMA_MAX","upper width for random proposals");
  keys.add("compulsory","RANDOM_TRIALS","1000","number of seeded random candidates before L-BFGS-B");
  keys.add("compulsory","MAX_ITER","15000","maximum local optimization iterations");
  keys.add("compulsory","TOL","0.000001","L-BFGS-B stopping tolerances");
  keys.add("compulsory","SEED","1","nonnegative random-search seed; saved with settings");
  keys.add("compulsory","INITIAL_PARAMS","1D: [a,b,sigma] blocks; 2D: [a,bx,by,sx,sy] blocks, each of NGAUSSIANS values");
  keys.add("optional","STATE_FILE","checkpoint path; default is label.state");
  keys.add("optional","HISTORY_FILE","complete per-window evidence path; default is label.history");
  keys.add("compulsory","CHECKPOINT_STRIDE","0","snapshot frequency in MD steps; zero means PACE; also saved at engine checkpoints and window boundaries");
  keys.addOutputComponent("mfpt","default","last selected model MFPT in active time units; -1 before first update");
  keys.addOutputComponent("window","default","number of completed bias windows before this force evaluation");
  keys.addOutputComponent("fpt","default","first detected physical target hit in active time units; -1 before a hit or when stopping is disabled");
}

Kombi::Kombi(const ActionOptions& ao):PLUMED_BIAS_INIT(ao),checkpointStride(0) {
  try {
    kombi::require(getNumberOfArguments()==1 || getNumberOfArguments()==2,"one or two scalar ARGs are supported");
    for(unsigned i=0;i<getNumberOfArguments();++i)kombi::require(getPntrToArgument(i)->getRank()==0 && !getPntrToArgument(i)->isPeriodic(),"each ARG must be a nonperiodic scalar");
    kombi::require(getStride()==1,"use SAMPLE_STRIDE for counting; bias STRIDE must remain 1");
    kombi::Settings c;c.dim=getNumberOfArguments();unsigned hills=10,lag=1,pace=10000,stride=1,trials=1000,maxIter=15000;unsigned seed=1;
    std::vector<double> low,high,target;std::vector<unsigned> bins;
    parseVector("GRID_MIN",low);parseVector("GRID_MAX",high);parseVector("TARGET",target);parseVector("BINS",bins);
    kombi::require(low.size()==c.dim && high.size()==c.dim && target.size()==c.dim && bins.size()==c.dim,"GRID_MIN, GRID_MAX, TARGET and BINS need one entry per ARG");
    c.low=low[0];c.high=high[0];c.target=target[0];c.bins=bins[0];
    if(c.dim==2){c.lowY=low[1];c.highY=high[1];c.targetY=target[1];c.binsY=bins[1];}
    std::string mode=c.dim==1?"RAS":"CENTERS";parse("GRID_MODE",mode);
    kombi::require(mode=="RAS"||mode=="CENTERS"||mode=="LEGACY_2D"||mode=="LEGACY_1D","unknown GRID_MODE");
    c.gridMode=mode=="RAS"?0:(mode=="CENTERS"?1:(mode=="LEGACY_2D"?2:3));parseFlag("CLIP_GRID",c.clipGrid);
    parseFlag("STOP_AT_TARGET",c.stopAtTarget);parse("TARGET_RADIUS",c.targetRadius);
    parse("KBT",c.kbt);c.tiltKbt=c.kbt;parse("TILT_KBT",c.tiltKbt);
    parse("NGAUSSIANS",hills);parse("LAG",lag);parse("PACE",pace);parse("SAMPLE_STRIDE",stride);
    parse("AMP_MIN",c.ampMin);parse("AMP_MAX",c.ampMax);parse("SIGMA_MIN",c.widthMin);parse("SIGMA_MAX",c.widthMax);
    parse("RANDOM_AMP",c.randomAmp);parse("RANDOM_SIGMA_MIN",c.randomWidthMin);parse("RANDOM_SIGMA_MAX",c.randomWidthMax);
    parse("RANDOM_TRIALS",trials);parse("MAX_ITER",maxIter);parse("TOL",c.tol);parse("SEED",seed);
    c.hills=hills;c.lag=lag;c.pace=pace;c.stride=stride;c.randomTrials=trials;c.maxIter=maxIter;c.seed=seed;c.validate();
    kombi::Vec initial;parseVector("INITIAL_PARAMS",initial);
    parse("CHECKPOINT_STRIDE",checkpointStride);if(checkpointStride==0)checkpointStride=pace;
    stateName=getLabel()+".state";historyName=getLabel()+".history";parse("STATE_FILE",stateName);parse("HISTORY_FILE",historyName);
    kombi::require(stateName!=historyName,"STATE_FILE and HISTORY_FILE must differ");checkRead();
    engine=std::make_unique<kombi::Engine>(c,initial);
    if(getRestart()) {
      IFile input;input.link(*this);input.open(stateName);std::string line,text;
      while(input.getline(line)) {text+=line+'\n';}
      input.close();engine->restore(text);
      // A restart appends history, which may already contain evidence beyond an older MD checkpoint.
      // Require an exact match; silently replaying old evidence would duplicate windows.
      IFile old;old.link(*this);old.open(historyName);std::size_t complete=0;
      std::string last;std::uint64_t digest=UINT64_C(14695981039346656037);
      while(old.getline(line)){if(line=="END_WINDOW")++complete;if(!line.empty())last=line;digest=kombi::checksum(line+'\n',digest);}
      old.close();kombi::require(complete==engine->windows && digest==engine->historyDigest && (complete==0||last=="END_WINDOW"),"history/checkpoint mismatch; restore the paired files from the same MD checkpoint");
    }
    state.link(*this);state.open(stateName);history.link(*this);history.open(historyName);
    if(!getRestart()) {
      const std::string header="KOMBI_HISTORY_V1\n"+c.signature()+"\n";
      history.printf("%s",header.c_str());history.flush();engine->historyDigest=kombi::checksum(header);
    }
    addComponent("mfpt");componentIsNotPeriodic("mfpt");mfptValue=getPntrToComponent("mfpt");
    addComponent("window");componentIsNotPeriodic("window");windowValue=getPntrToComponent("window");
    addComponent("fpt");componentIsNotPeriodic("fpt");fptValue=getPntrToComponent("fpt");
    log.printf("  directed hard-bin KOMBI: %zu dimensions, %zu states, %s grid, %u positive Gaussians; PACE=%u, SAMPLE_STRIDE=%u, LAG=%u\n",c.dim,c.states(),mode.c_str(),hills,pace,stride,lag);
    log.printf("  DHAM KBT=%.12g, tilt KBT=%.12g; all input quantities use active PLUMED units\n",c.kbt,c.tiltKbt);
  } catch(const std::exception& e) {error(e.what());}
}
void Kombi::calculate() {
  try {
    const double x=getArgument(0),y=engine->cfg.dim==2?getArgument(1):0;
    if(!engine->cfg.hitsTarget(x,y))engine->cfg.state(x,y);
    auto v=kombi::gaussianND(x,y,engine->appliedAt(getStep()),engine->cfg.dim);setBias(v[0]);setOutputForce(0,-v[1]);
    if(engine->cfg.dim==2)setOutputForce(1,-v[2]);
    const bool replay=getStep()==engine->lastStep;
    mfptValue->set(replay?engine->replayMfpt:engine->lastMfpt);windowValue->set(replay?engine->replayWindows:engine->windows);
    const auto hit=engine->firstPassageStep>=0?engine->firstPassageStep:(engine->cfg.hitsTarget(x,y)?static_cast<long long>(getStep()):-1);
    fptValue->set(hit<0?-1:hit*getTimeStep());
  } catch(const std::exception& e) {error(e.what());}
}
void Kombi::update() {
  try {
    kombi::Decision d;const bool changed=engine->observe(getStep(),getArgument(0),getTimeStep(),d,engine->cfg.dim==2?getArgument(1):0);
    if(changed)record(d);
    if(changed || getCPT() || getStep()%checkpointStride==0 || engine->firstPassageStep>=0)checkpoint();
    if(engine->firstPassageStep>=0)plumed.stop();
  } catch(const std::exception& e) {
    // Save the failed window separately. Never replace the last usable restart with failed state.
    OFile failure;failure.link(*this);failure.open(stateName+".failed");failure.printf("%s\n%s",e.what(),engine->save().c_str());failure.flush();error(e.what());
  }
}
}}
