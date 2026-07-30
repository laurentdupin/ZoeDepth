#version 450 core
layout(local_size_x=256) in;
layout(set=0,binding=0,std430) writeonly buffer O{float v[];}o;
layout(set=0,binding=1,std430) readonly buffer I{float v[];}i;
layout(push_constant) uniform P{uint spatial;uint features;}p;
void main(){
 uint id=gl_GlobalInvocationID.x,t=id/p.features,f=id%p.features;
 if(t>p.spatial||f>=p.features)return;
 uint pair=f%(p.features/2);
 float angle=float(t)*exp(float(pair*2)*(-9.210340371976184/float(p.features)));
 float x=f<p.features/2?sin(angle):cos(angle);
 if(t>0)x+=i.v[f*p.spatial+t-1];
 o.v[id]=x;
}
