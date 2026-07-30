#version 450 core
layout(local_size_x=64) in;
layout(set=0,binding=0,std430) writeonly buffer N{float v[];}next_centers;
layout(set=0,binding=1,std430) writeonly buffer O{float v[];}output_centers;
layout(set=0,binding=2,std430) readonly buffer P{float v[];}previous;
layout(set=0,binding=3,std430) readonly buffer A{float v[];}attractors;
layout(push_constant) uniform Params{uint pixels;uint bins;uint count;uint normed;}p;
void main(){
 uint pixel=gl_GlobalInvocationID.x;if(pixel>=p.pixels)return;
 float sorted[64];
 for(uint b=0;b<p.bins;++b){
  float center=previous.v[b*p.pixels+pixel],delta=0.0;
  for(uint a=0;a<p.count;++a){
   uint channel=p.normed!=0?a*2:a;
   float d=attractors.v[channel*p.pixels+pixel]-center;
   delta+=d/(1.0+1000.0*d*d);
  }
  float updated=center+delta/float(p.count);
  next_centers.v[b*p.pixels+pixel]=updated;
  sorted[b]=p.normed!=0?clamp(9.999*updated+.001,.001,10.0):updated;
 }
 if(p.normed!=0){
  for(uint a=1;a<p.bins;++a){float x=sorted[a];int j=int(a)-1;while(j>=0&&sorted[j]>x){sorted[j+1]=sorted[j];--j;}sorted[j+1]=x;}
 }
 for(uint b=0;b<p.bins;++b)output_centers.v[b*p.pixels+pixel]=sorted[b];
}
