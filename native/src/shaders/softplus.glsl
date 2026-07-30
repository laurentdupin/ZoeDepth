#version 450 core
layout(local_size_x=256) in;
layout(set=0,binding=0,std430) buffer V{float v[];}values;
layout(push_constant) uniform P{uint count;}p;
void main(){uint i=gl_GlobalInvocationID.x;if(i<p.count){float x=values.v[i];values.v[i]=log(1.0+exp(-abs(x)))+max(x,0.0);}}
