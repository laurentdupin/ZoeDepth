#version 450 core
layout(local_size_x=256) in;
layout(set=0,binding=0,std430) writeonly buffer O{float v[];}o;
layout(set=0,binding=1,std430) readonly buffer I{float v[];}i;
layout(push_constant) uniform P{uint tokens;uint channels;}p;
void main(){
 uint id=gl_GlobalInvocationID.x;
 if(id>=p.tokens*p.channels)return;
 uint t=id/p.channels,c=id%p.channels;
 o.v[c*p.tokens+t]=i.v[id];
}
