#version 450 core
layout(local_size_x=256) in;
layout(set=0,binding=0,std430) buffer S{float v[];}scores;
layout(set=0,binding=1,std430) readonly buffer T{float v[];}table;
layout(push_constant) uniform P{uint pw;uint ph;uint tokens;uint heads;}p;
float table_at(uint x,uint y,uint h){
 return table.v[(x*47+y)*p.heads+h];
}
float interpolate(uint x,uint y,uint rw,uint rh,uint h){
 float sx=clamp((float(x)+.5)*47.0/float(rw)-.5,0.0,46.0);
 float sy=clamp((float(y)+.5)*47.0/float(rh)-.5,0.0,46.0);
 uint x0=uint(floor(sx)),y0=uint(floor(sy));
 uint x1=min(x0+1,46u),y1=min(y0+1,46u);
 return mix(mix(table_at(x0,y0,h),table_at(x1,y0,h),sx-float(x0)),
            mix(table_at(x0,y1,h),table_at(x1,y1,h),sx-float(x0)),
            sy-float(y0));
}
void main(){
 uint id=gl_GlobalInvocationID.x,total=p.heads*p.tokens*p.tokens;
 if(id>=total)return;
 uint k=id%p.tokens,q=(id/p.tokens)%p.tokens,h=id/(p.tokens*p.tokens);
 uint rw=2*p.pw-1,rh=2*p.ph-1,spatial=rw*rh,index;
 float bias;
 if(q==0&&k==0)bias=table.v[(2209+2)*p.heads+h];
 else if(q==0)bias=table.v[2209*p.heads+h];
 else if(k==0)bias=table.v[(2209+1)*p.heads+h];
 else{
  uint qp=q-1,kp=k-1;
  int dy=int(qp/p.pw)-int(kp/p.pw);
  int dx=int(qp%p.pw)-int(kp%p.pw);
  index=uint((dy+int(p.ph)-1)*int(rw)+dx+int(p.pw)-1);
  uint x=index/rh,y=index%rh;
  bias=interpolate(x,y,rw,rh,h);
 }
 scores.v[id]+=bias;
}
