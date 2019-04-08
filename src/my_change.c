#include"my_change.h"


void Print_net(network* net) {

	for (int i = 0; i < net->n-1; i++) {
		layer l = net->layers[i];
		LAYER_TYPE T = l.type;
	}
}

void mall_ptr_array(float * sized, float* mall_ptr) {
	for (int i = 0; i < _msize(mall_ptr)/sizeof(float); i++) {
			sized[i] = mall_ptr[i];
	}
	return;
}
int print_local(float* data)
{
	FILE *fp;

	fp = fopen("out_c.data", "w");
	if (fp == NULL)
	{
		printf("File cannot open! ");
		exit(0);
	}
	int i = 0;
	for (i = 0; i <_msize(data)/sizeof(float); i++) {
		//printf("%d, %f\n ",i, data[i]);
		fprintf(fp,"%f ",data[i]);
	}
	fclose(fp);
	return 0;
}
void l0_forward_convolutional_layer(convolutional_layer l, network_state state)
{/*
    l.outputs = l.out_h * l.out_w * l.out_c;某层计算之后输出的参数数量
    l.inputs = l.w * l.h * l.c;层的输入参数数量
    l.output = calloc(l.batch*l.outputs, sizeof(float));还要乘以batch,不过这里都是1
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));这是反向计算时对于权值的差值。
 */
	int out_h = convolutional_out_height(l);
	int out_w = convolutional_out_width(l);
	int i;
	fill_cpu(l.outputs*l.batch, 0, l.output, 1);//输出结果数组的初始化。
	int m = l.n;
	int k = l.size*l.size*l.c;
	int n = out_h*out_w;
	float *a = l.weights;//l.weights = calloc(c*n*size*size, sizeof(float));n是卷积核个数
	float *b = state.workspace;//l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);保存的都是float类型。
	float *c = l.output;
	static int u = 0;
	u++;
	//上边的b只是开辟了一段足够大的空间，来存放输入数据。
	//然后下面的函数对b又进行了修改，将输入数据放进去。比输入数据多的那部分为0,
	im2col_cpu_custom(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
	//print_local(a);
	l0_gemm_cpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
	print_local(c);
	c += n*m;
	state.input += l.c*l.h*l.w;

	/*  if(l.batch_normalize){
	forward_batchnorm_layer(l, state);
	}*/
	add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);

	//activate_array(l.output, m*n*l.batch, l.activation);
	activate_array_cpu_custom(l.output, m*n*l.batch, l.activation);

	//if(l.binary || l.xnor) swap_binary(&l);
}

//void gemm_nn(int M, int N, int K, float ALPHA,
//	float *A, int lda,
//	float *B, int ldb,
//	float *C, int ldc)
//{
//	int ta = _msize(A) / sizeof(float);
//	int tb = _msize(B) / sizeof(float);
//	int tc = _msize(C) / sizeof(float);
//	int i, j, k;
//	if (is_avx() == 1) {    // AVX
//		for (i = 0; i < M; ++i) {
//			for (k = 0; k < K; ++k) {
//				float A_PART = ALPHA*A[i*lda + k];
//				__m256 a256, b256, c256, result256;    // AVX
//				a256 = _mm256_set1_ps(A_PART);
//				for (j = 0; j < N - 8; j += 8) {
//					b256 = _mm256_loadu_ps(&B[k*ldb + j]);
//					c256 = _mm256_loadu_ps(&C[i*ldc + j]);
//					// FMA - Intel Haswell (2013), AMD Piledriver (2012)
//					//result256 = _mm256_fmadd_ps(a256, b256, c256);
//					result256 = _mm256_mul_ps(a256, b256);
//					result256 = _mm256_add_ps(result256, c256);
//					_mm256_storeu_ps(&C[i*ldc + j], result256);
//				}
//
//				int prev_end = (N % 8 == 0) ? (N - 8) : (N / 8) * 8;
//				for (j = prev_end; j < N; ++j)
//					C[i*ldc + j] += A_PART*B[k*ldb + j];
//			}
//		}
//	}
//	else {
//		for (i = 0; i < M; ++i) {
//			for (k = 0; k < K; ++k) {
//				register float A_PART = ALPHA*A[i*lda + k];
//				for (j = 0; j < N; ++j) {
//					C[i*ldc + j] += A_PART*B[k*ldb + j];
//				}
//				/* // SSE
//				__m128 a128, b128, c128, result128;    // SSE
//				a128 = _mm_set1_ps(A_PART);
//				for (j = 0; j < N - 4; j += 4) {
//				b128 = _mm_loadu_ps(&B[k*ldb + j]);
//				c128 = _mm_loadu_ps(&C[i*ldc + j]);
//				//result128 = _mm_fmadd_ps(a128, b128, c128);
//				result128 = _mm_mul_ps(a128, b128);
//				result128 = _mm_add_ps(result128, c128);
//				_mm_storeu_ps(&C[i*ldc + j], result128);
//				}
//
//				int prev_end = (N % 4 == 0) ? (N - 4) : (N / 4) * 4;
//				for (j = prev_end; j < N; ++j){
//				C[i*ldc + j] += A_PART*B[k*ldb + j];
//				}
//				*/
//			}
//		}
//	}
//}

