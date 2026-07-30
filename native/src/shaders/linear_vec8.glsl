#version 450 core
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(set=0,binding=0,std430) writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430) readonly buffer I{vec4 d[];}i;
layout(set=0,binding=2,std430) readonly buffer W{vec4 d[];}w;
layout(set=0,binding=3,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{uint rows;uint inputs;uint outputs;}p;
#define K 8
shared vec4 it[32*K];
shared vec4 wt[32*K];
void main(){
 uint cb=gl_WorkGroupID.x*32+gl_LocalInvocationID.x*4;
 uint rb=gl_WorkGroupID.y*32+gl_LocalInvocationID.y*4;
 float s[4][4];
 for(uint r=0;r<4;++r)for(uint c=0;c<4;++c)s[r][c]=0.0;
 uint lane=gl_LocalInvocationID.y*8+gl_LocalInvocationID.x;
 uint iv=p.inputs/4;
 for(uint base=0;base<iv;base+=K){
  for(uint n=lane;n<32*K;n+=64){
   uint tr=n/K,inner=base+n%K,row=gl_WorkGroupID.y*32+tr;
   it[n]=row<p.rows&&inner<iv?i.d[row*iv+inner]:vec4(0.0);
  }
  for(uint n=lane;n<32*K;n+=64){
   uint tc=n/K,inner=base+n%K,col=gl_WorkGroupID.x*32+tc;
   wt[n]=col<p.outputs&&inner<iv?w.d[col*iv+inner]:vec4(0.0);
  }
  barrier();
  uint count=min(K,iv-base);
  for(uint inner=0;inner<count;++inner){
   vec4 a[4],q[4];
   for(uint r=0;r<4;++r)a[r]=it[(gl_LocalInvocationID.y*4+r)*K+inner];
   for(uint c=0;c<4;++c)q[c]=wt[(gl_LocalInvocationID.x*4+c)*K+inner];
   for(uint r=0;r<4;++r)for(uint c=0;c<4;++c)s[r][c]+=dot(a[r],q[c]);
  }
  barrier();
 }
 for(uint r=0;r<4;++r){
  uint row=rb+r;if(row>=p.rows)continue;
  for(uint c=0;c<4;++c){uint col=cb+c;if(col<p.outputs)
   o.d[row*p.outputs+col]=s[r][c]+b.d[col];}
 }
}
