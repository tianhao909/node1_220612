#include "/usr/local/cuda-11.3/targets/x86_64-linux/include/cuda_runtime.h"
#include<stdlib.h>
#include <stdio.h>
#include <assert.h>


void cpu(int *a, int N) {
    for(int i=0; i<N; i++){
        a[i] = i;
    }
    // int *a =  (int* )malloc(sizeof(int));
    // free(a);
    // printf("hello cpu\n");
    
}

//kernel 函数 ， 没有返回值
__global__ void gpu(int *a , int N){  //标志 函数在gpu上运行  必须返回void  
    int threadi = blockIdx.x * blockDim.x + threadIdx.x;
    
    //int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x; //gridDim.x 一个网格有几个block； blockDim 一个block有几个线程； 跨步stride 一个网格有几个线程
    //printf("gridDim.x=%d,blockDim.x=%d, stride=%d \n",gridDim.x,blockDim.x,stride);
    for (int i = threadi; i<N; i += stride) {//每次增加 stride个
        //printf("i=%d \n", i);
        a[i] *= 2;
        //printf("a[%d] = %d \n", i, a[i]);
    }
}

bool check(int *a, int N){
    for(int i=0; i<N; i++){
        //printf("true: %d\n",i);
        if(a[i] != 2*i){
            //printf("false: %d\n",i);
            return false;
        }
        
        return true;
    }
}

inline cudaError_t checkCuda(cudaError_t result){
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);//// assert 的作用是现计算表达式 expression     ，如果其值为假（即为0），那么它先向 stderr 打印一条出错信息,然后通过调用 abort >    来终止程序运行。void assert( int expression );  #include "assert.h" 
    }
    return result;
}

int main(){
    //const int N = 2 << 20;
    //const int N  = 100;
    const int N  = 1000;
    size_t size = N * sizeof(int);
    //int *a = (int *)malloc(size); 
    int *a;
    cudaError_t err;
    err = cudaMallocManaged(&a, size); // 分配的是一个统一的内存，既可以被cpu 也可以被 gpu使用 cpu函数 不能放到 __global__ void gpu()函数中运行
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    cpu(a, N);

    size_t threads = 256; //每个blocks的线程数
    size_t blocks = 1;
    err = cudaGetLastError(); //判断 __global__ void gpu(int *a , int N){  //标志 函数在gpu上运行  必须返回void   是否报错
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    //size_t blocks = (N + threads - 1) / threads; //上取整  让N个数 每一个数 都有一个线程 
    gpu<<<blocks, threads>>>(a, N); // 启动函数操作， 必须三个<<< >>>   1,1 线程块的个数  线程个数
     
    checkCuda(cudaDeviceSynchronize());//cpu等待gpu代码执行完成
    check(a,N) ? printf("ok") : printf("error");
    cudaFree(a);
    //free(a);

    
    
    
}
