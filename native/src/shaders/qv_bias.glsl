#version 450 core
layout(local_size_x=256) in;
layout(set=0,binding=0,std430) buffer Q{float v[];}q;
layout(set=0,binding=1,std430) readonly buffer QB{float v[];}qb;
layout(set=0,binding=2,std430) readonly buffer VB{float v[];}vb;
layout(push_constant) uniform P{uint tokens;uint embedding;}p;
void main(){
 uint id=gl_GlobalInvocationID.x;
 if(id>=p.tokens*p.embedding)return;
 uint t=id/p.embedding,c=id%p.embedding,base=t*3*p.embedding;
 q.v[base+c]+=qb.v[c]; q.v[base+2*p.embedding+c]+=vb.v[c];
}
