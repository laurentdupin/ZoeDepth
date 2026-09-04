#version 450 core
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(set=0,binding=0,std430) writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430) readonly buffer I{float d[];}i;
layout(set=0,binding=2,std430) readonly buffer W{float d[];}w;
layout(set=0,binding=3,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{
 uint iw;uint ih;uint ic;uint ow;uint oh;uint oc;
 uint kernel;uint stride;int padding;uint has_bias;
 uint batches;uint channel_blocks;
}p;
shared float st[100];
shared vec4 kt[9];
void main(){
 uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y;
 uint block=gl_GlobalInvocationID.z%p.channel_blocks;
 uint batch=gl_GlobalInvocationID.z/p.channel_blocks;
 uint cb=block*4;
 bool valid=x<p.ow&&y<p.oh&&cb<p.oc&&batch<p.batches;
 vec4 s=vec4(0.0);
 uint lane=gl_LocalInvocationID.y*8+gl_LocalInvocationID.x;
 int ox=int(gl_WorkGroupID.x*8)-1,oy=int(gl_WorkGroupID.y*8)-1;
 uint input_batch=batch*p.ic*p.ih*p.iw;
 uint output_batch=batch*p.oc*p.oh*p.ow;
 for(uint c=0;c<p.ic;++c){
  for(uint n=lane;n<100;n+=64){
   int px=ox+int(n%10),py=oy+int(n/10);
   st[n]=px>=0&&px<int(p.iw)&&py>=0&&py<int(p.ih)
    ?i.d[input_batch+(c*p.ih+uint(py))*p.iw+uint(px)]:0.0;
  }
  if(lane<9){
   vec4 weights=vec4(0.0);
   for(uint off=0;off<4;++off){
    uint ch=cb+off;
    if(ch<p.oc)weights[off]=w.d[(ch*p.ic+c)*9+lane];
   }
   kt[lane]=weights;
  }
  barrier();
  if(valid)for(uint ky=0;ky<3;++ky)for(uint kx=0;kx<3;++kx){
   float v=st[(gl_LocalInvocationID.y+ky)*10+
              gl_LocalInvocationID.x+kx];
   s+=v*kt[ky*3+kx];
  }
  barrier();
 }
 if(!valid)return;
 for(uint off=0;off<4;++off){uint ch=cb+off;if(ch<p.oc)
  o.d[output_batch+(ch*p.oh+y)*p.ow+x]=
   s[off]+(p.has_bias!=0?b.d[ch]:0.0);}
}
