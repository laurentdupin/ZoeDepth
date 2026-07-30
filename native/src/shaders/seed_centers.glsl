#version 450 core
layout(local_size_x=256) in;
layout(set=0,binding=0,std430) writeonly buffer O{float v[];}o;
layout(set=0,binding=1,std430) readonly buffer I{float v[];}i;
layout(push_constant) uniform P{uint pixels;uint bins;}p;
void main(){
 uint pixel=gl_GlobalInvocationID.x;if(pixel>=p.pixels)return;
 float sum=0.0;
 for(uint b=0;b<p.bins;++b)sum+=max(i.v[b*p.pixels+pixel],0.0)+.001;
 float edge=0.0;
 for(uint b=0;b<p.bins;++b){
  float width=(max(i.v[b*p.pixels+pixel],0.0)+.001)/sum;
  o.v[b*p.pixels+pixel]=edge+.5*width;edge+=width;
 }
}
