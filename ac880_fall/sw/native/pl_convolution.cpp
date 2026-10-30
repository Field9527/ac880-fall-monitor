// NCNN custom pointwise layer. Ordinary convolution/depthwise layers stay in NCNN.
// BSD-3-Clause; project-owned adapter, links to Tencent NCNN.
#include "net.h"
#include "layer.h"
#include "modelbin.h"
#include <cstdint>

typedef int (*compute_fn)(const float*,const float*,const float*,float*,
    unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned);
static compute_fn compute_callback=nullptr;
static unsigned long long calls=0,macs=0;

class PLPointwise : public ncnn::Layer {
public:
    PLPointwise(){one_blob_only=true;support_inplace=false;support_packing=false;}
    int load_param(const ncnn::ParamDict& pd) override {
        co=pd.get(0,0);int kw=pd.get(1,0),kh=pd.get(11,kw);
        int sw=pd.get(3,1),sh=pd.get(13,sw),pad=pd.get(4,0);
        bias_term=pd.get(5,0);int count=pd.get(6,0);relu=pd.get(9,0);
        if(kw!=1||kh!=1||sw!=1||sh!=1||pad!=0||co<1||co>960||count%co||relu<0||relu>1)return -310;
        ci=count/co;
        if(ci<1||ci>960||bias_term<0||bias_term>1)return -311;
        return 0;
    }
    int load_model(const ncnn::ModelBin& mb) override {
        weights=mb.load(ci*co,0);
        if(weights.empty()||weights.elemsize!=4u)return -312;
        if(bias_term){bias=mb.load(co,1);if(bias.empty())return -313;}
        return 0;
    }
    int forward(const ncnn::Mat& bottom,ncnn::Mat& top,const ncnn::Option& opt) const override {
        if(!compute_callback||bottom.dims!=3||bottom.c!=ci||bottom.elempack!=1||bottom.elemsize!=4u)return -314;
        top.create(bottom.w,bottom.h,co,(size_t)4u,1,opt.blob_allocator);
        if(top.empty())return -100;
        unsigned pixels=bottom.w*bottom.h;
        int rc=compute_callback((const float*)bottom.data,(const float*)weights.data,
            bias_term?(const float*)bias.data:nullptr,(float*)top.data,
            ci,co,pixels,bottom.cstep,top.cstep,relu,bias_term);
        if(rc==0){calls++;macs+=(unsigned long long)ci*co*pixels;}
        return rc;
    }
private:
    int ci=0,co=0,relu=0,bias_term=0;
    ncnn::Mat weights,bias;
};
static ncnn::Layer* create(void*){return new PLPointwise;}
static void destroy(ncnn::Layer* l,void*){delete l;}

extern "C" {
int ac880_register_pl(void* net){
    // C API ncnn_net_t contains the C++ Net pointer as its first field.
    return ((ncnn::Net*)*(void**)net)->register_custom_layer("PLPointwise",create,destroy,nullptr);
}
void ac880_set_compute_callback(compute_fn callback){compute_callback=callback;}
unsigned long long ac880_pl_calls(){return calls;}
unsigned long long ac880_pl_macs(){return macs;}
}
