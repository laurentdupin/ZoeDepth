#version 450 core
layout(local_size_x=256) in;
layout(set=0,binding=0,std430) writeonly buffer Q{float v[];}q;
layout(set=0,binding=1,std430) writeonly buffer K{float v[];}k;
layout(set=0,binding=2,std430) writeonly buffer V{float v[];}v;
layout(set=0,binding=3,std430) readonly buffer I{float v[];}i;
layout(push_constant) uniform P{uint tokens;uint dimensions;}p;
void main(){uint id=gl_GlobalInvocationID.x;if(id>=p.tokens*p.dimensions)return;uint t=id/p.dimensions,c=id%p.dimensions,b=t*3*p.dimensions;q.v[id]=i.v[b+c];k.v[id]=i.v[b+p.dimensions+c];v.v[id]=i.v[b+2*p.dimensions+c];}
