#version 450 core
layout(local_size_x=256) in;
layout(set=0,binding=0,std430) writeonly buffer O{float v[];}o;
layout(set=0,binding=1,std430) readonly buffer I{float v[];}i;
layout(push_constant) uniform P{uint patches;uint embedding;}p;
void main(){
 uint id=gl_GlobalInvocationID.x;
 if(id>=p.patches*p.embedding*2)return;
 uint row=id/(p.embedding*2),c=id%(p.embedding*2);
 o.v[id]=c<p.embedding?i.v[(row+1)*p.embedding+c]:i.v[c-p.embedding];
}
