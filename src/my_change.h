#ifndef MY_CHANGE_H_
#define MY_CHANGE_H_
#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include "network.h"
#include<malloc.h>

//一定要把这个头文件包含进来，里边包含了gemm 和 im2col_cpu_custom这两个函数。
//会用在l0_forward_convolutional_layer这个函数中，然后my_change.c也会把这两个函数包含进去。链接才不会出错。
#include<immintrin.h>
void Print_net(network* net);
void mall_ptr_array(float *, float*);
void l0_forward_convolutional_layer(convolutional_layer l, network_state state);


#endif//MY_CHANGE_H_