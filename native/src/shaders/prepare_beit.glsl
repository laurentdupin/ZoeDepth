#version 450 core
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set=0,binding=0,std430) writeonly buffer O{float v[];}o;
layout(set=0,binding=1,std430) readonly buffer I{float v[];}i;
layout(set=0,binding=2,std430) readonly buffer W{float v[];}w;
layout(set=0,binding=3,std430) readonly buffer B{float v[];}b;
layout(set=0,binding=4,std430) readonly buffer C{float v[];}c;
layout(push_constant) uniform P{uint width;uint height;uint pw;uint ph;}p;
void main(){
 uint f=gl_GlobalInvocationID.x,t=gl_GlobalInvocationID.y;
 uint tokens=p.pw*p.ph+1;
 if(f>=1024||t>=tokens)return;
 if(t==0){o.v[f]=c.v[f];return;}
 uint id=t-1,px=id%p.pw,py=id/p.pw; float s=b.v[f];
 for(uint ch=0;ch<3;++ch)for(uint y=0;y<16;++y)for(uint x=0;x<16;++x)
  s+=i.v[(ch*p.height+py*16+y)*p.width+px*16+x]*
    w.v[((f*3+ch)*16+y)*16+x];
 o.v[t*1024+f]=s;
}
