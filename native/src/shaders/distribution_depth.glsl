#version 450 core
layout(local_size_x=64) in;
layout(set=0,binding=0,std430) writeonly buffer O{float v[];}depth;
layout(set=0,binding=1,std430) readonly buffer P{float v[];}parameters;
layout(set=0,binding=2,std430) readonly buffer C{float v[];}centers;
layout(push_constant) uniform Params{uint pixels;}p;
float log_binomial(uint bin){
 if(bin==0||bin==63)return 0.0;
 float k=float(bin),n=63.0,nk=n-k;
 return n*log(n)-k*log(k)-nk*log(nk);
}
void main(){
 uint pixel=gl_GlobalInvocationID.x;if(pixel>=p.pixels)return;
 const float e=.0001;
 float p0=parameters.v[pixel]+e,p1=parameters.v[p.pixels+pixel]+e;
 float probability=p0/(p0+p1);
 float t0=parameters.v[2*p.pixels+pixel]+e,t1=parameters.v[3*p.pixels+pixel]+e;
 float temperature=(50.0-.0212)*(t0/(t0+t1))+.0212;
 float maximum=-3.402823466e38;
 for(uint b=0;b<64;++b){
  float k=float(b);
  float comb=log_binomial(b);
  float q=clamp(1.0-probability,e,1.0),pp=clamp(probability,e,1.0);
  float logit=(comb+k*log(pp)+(63.0-k)*log(q))/temperature;
  maximum=max(maximum,logit);
 }
 float denominator=0.0;
 for(uint b=0;b<64;++b){
  float k=float(b);
  float comb=log_binomial(b);
  float q=clamp(1.0-probability,e,1.0),pp=clamp(probability,e,1.0);
  float logit=(comb+k*log(pp)+(63.0-k)*log(q))/temperature;
  denominator+=exp(logit-maximum);
 }
 float value=0.0;
 for(uint b=0;b<64;++b){
  float k=float(b);
  float comb=log_binomial(b);
  float q=clamp(1.0-probability,e,1.0),pp=clamp(probability,e,1.0);
  float logit=(comb+k*log(pp)+(63.0-k)*log(q))/temperature;
  value+=exp(logit-maximum)/denominator*centers.v[b*p.pixels+pixel];
 }
 depth.v[pixel]=value;
}
