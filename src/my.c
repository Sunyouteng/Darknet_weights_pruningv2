#include<stdlib.h >
#include<malloc.h>
#include<math.h>
#include"my.h"
//#include"my.h"
_Bool check_next_layer(network * net, int i, int nth_layer)
{/*
 函数功能：检查当前层i是否是需要裁减层的下一层
 参数说明： i	某层的索引
			nth_layer	待裁减层索引
返回值	 ：0 不是， 1 是。
 */
	int flag = 0;
	if (i <= nth_layer)
		return 0;
	else
	{
		for (int m = nth_layer+1; m <= i;m++)
		{
			if (net->layers[m].type == CONVOLUTIONAL)
				flag++;
		}
	}
	if (flag == 1)
		return 1;
	else
		return 0;

}
void * safe_malloc( int size)
{
	void* filters_sum = malloc(size);
	if (filters_sum == NULL)
		perror("error, malloc failed...");
	memset(filters_sum, 0, size);
	return filters_sum;
}

void quick_sort(struct stu* s, int l, int r)
{
	if (l < r)
	{
		int i = l, j = r;

		struct stu  x = s[l];
		while (i<j)
		{
			while (i<j && s[j].value >= x.value)//从右到左找到第一个小于x的数  
				j--;
			if (i<j)
				s[i++] = s[j];

			while (i<j && s[i].value <= x.value)//从左往右找到第一个大于x的数  
				i++;
			if (i<j)
				s[j--] = s[i];
		}
		s[i] = x;//i = j的时候，将x填入中间位置  
		quick_sort(s, l, i - 1);//递归调用 
		quick_sort(s, i + 1, r);
	}
}

int* sort_filters(float * weights, float *bias, int filter_number, int prune_filters)
{/*
 功能：对每个卷积核和偏倚取绝对值求和，得到以每个和作为指标评价卷积核的重要性。
 参数说明：weights 权重数组 bias数组 filter_number卷积核数目 prune_filters要剪掉的卷积核数目
 */
	//filters_sum里保存了每个卷积核的权值和偏倚的绝对值的和。
	float* filters_sum = (float*)safe_malloc(sizeof(float)* filter_number);

	int* result = (int*)safe_malloc(sizeof(int)*prune_filters);

//卷积核求和
	for (int i = 0; i < _msize(weights) / sizeof(float); i++)
	{
		filters_sum[i%filter_number] += fabsf(weights[i]);
	}
	//printf("before sort filters_sum are:\n");
	for (int i = 0; i < filter_number; i++)
	{
		filters_sum[i] += fabsf(bias[i]);
		//printf("%f\n",filters_sum[i]);
	}
	//对filters_sum进行排序，得到prune_filters个最小的卷积核的编号。
	struct stu* filters_sum_index = (struct stu*)safe_malloc(sizeof(struct stu)*filter_number);

	for (int i = 0; i < filter_number; i++)
	{
		filters_sum_index[i].index = i;
		filters_sum_index[i].value = filters_sum[i];
	}
	quick_sort(filters_sum_index, 0, filter_number-1);
	printf("after sort  filter's indexs are:\n");
	//for (int i = 0; i < filter_number; i++)
	//{
	//	printf("%f\n", filters_sum_index[i].value);
	//}
	printf("after sort the least %d filter's indexs are:\n", prune_filters);
	for (int i = 0; i < prune_filters; i++)
	{
			 result[i] = filters_sum_index[i].index;
			 printf(" %d\n", result[i]);
	}
	free(filters_sum_index);
	free(filters_sum);
	return result;
}
int get_channel_index(int weight_index,int kernel_size,int channel_size,int n)
{
	int result = weight_index % (kernel_size*kernel_size*channel_size) / (kernel_size*kernel_size);
	return result;
}
_Bool check_index(int i, int size_per_filter, int* prune_filters_index)
{/*
 功能：检查权值的索引是否属于被剪掉的卷积核
 */
	int flag = 0;
	int index_num = _msize(prune_filters_index) / sizeof(int);
	for (int j = 0; j < index_num; j++)
	{
		if ((int)(i / size_per_filter) != prune_filters_index[j])
			continue;
		else
			flag++;
	}
	if (flag)
		return 0;
	else
		return 1;
}
_Bool check_index_channel(int channel, int* prune_filters_index)
{/*
 功能：检查权值的索引是否属于卷积核被剪掉的通道
 */
	int flag = 0;
	int index_num = _msize(prune_filters_index) / sizeof(int);
	for (int j = 0; j < index_num; j++)
	{
		if (channel != prune_filters_index[j])
			continue;
		else
			flag++;
	}
	if (flag)
		return 0;
	else
		return 1;
}

float* prune_weights(float * weights, int* prune_filters_index, int size_per_filter, int filters_left)
{/*
形参说明：weights初始权值  prune_filters_index 需要修改的卷积核索引 
			size_per_filter 每个卷积核元素个数   filters_left 裁减后保留多少个卷积核
返回说明：修剪之后的权值地址	
 */			
	int weights_elements_num = _msize(weights) / sizeof(float);
	int index_num = _msize(prune_filters_index) / sizeof(int);
	float *new_weights = (float*)safe_malloc(sizeof(float)*size_per_filter* filters_left);
	int m = 0;
	for (int i = 0; i < weights_elements_num; i++)
	{
		if (check_index(i, size_per_filter, prune_filters_index))
			new_weights[m++] = weights[i];
		else
			continue;
	}
	free(weights);//这是新增的，实验一下。
	return new_weights;
}
float* prune_weights_nextlayer(float * weights, int kernel_size, int channel_size,int n, int* channel_index)
{/*
 功能：对被动裁减卷积层减去相应的通道
 形参说明：weights初始权值  kernel_size 卷积核尺寸  channel_size卷积核通道数
		   n 该层卷积核个数	channel_index 待修剪的通道数
 返回说明：修剪之后的权值地址
 */
	int weights_elements_num = _msize(weights) / sizeof(float);
	int index_num = _msize(channel_index) / sizeof(int);
	int size_per_filter = kernel_size*kernel_size*channel_size;
	int size_per_filter_after_prune = kernel_size*kernel_size*(channel_size - index_num);
	float *new_weights = (float*)safe_malloc(sizeof(float)*size_per_filter_after_prune* n);
	int m = 0;

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < channel_size; j++)
		{
			for (int k = 0; k < kernel_size*kernel_size;k++) {//对每个原来的权值的索引进行判断
				int index = (i*size_per_filter)+j*kernel_size*kernel_size + k;
				int channel = get_channel_index(index, kernel_size, channel_size, n);
				if (check_index_channel(channel, channel_index))
					new_weights[m++] = weights[index];
				else
					continue;
			}
		}
	}
	free(weights);
	return new_weights;
}
_Bool check_bias_index(int i, int* prune_filters_index)
{
	int flag = 0;
	int index_num = _msize(prune_filters_index) / sizeof(int);
	for (int j = 0; j < index_num; j++)
	{
		if (i != prune_filters_index[j])
			continue;
		else
			flag++;
	}
	if (flag)
		return 0;
	else
		return 1;
}
float* prune_bias(float * bias, int* prune_filters_index,int filters_left)
{//参数说明： bias原始偏倚数组 prune_filters_index 需要剪掉的偏倚索引 filters_left剩余偏倚的个数，
//				用于申请新的偏倚数组内存
//返回参数：裁减之后的偏倚数组地址
	int bias_elements_num = _msize(bias) / sizeof(float);
	int index_num = _msize(prune_filters_index) / sizeof(int);
	float *new_bias = (float*)safe_malloc(sizeof(float)* filters_left);
	int m = 0;
	for (int i = 0; i < bias_elements_num; i++)
	{
		if (check_bias_index(i,  prune_filters_index))
			new_bias[m++] = bias[i];
		else
			continue;
	}
	free(bias);
	return new_bias;
}