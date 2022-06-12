#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void square(float* d_out,float* d_in){ // 下划线 gpu运行  square是kernel函数   
  int idx = threadIdx.x; 
  float f = d_in[idx]; 
  printf("f = %f \n",f);
  d_out[idx] = f * f; //求平方  算出d-out是在GPU上
}

int main(int argc,char** argv){ 
  const int ARRAY_SIZE = 8; //分配空间 定义数组size  8位数组
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float); 

  // generate the input array on the host
  float h_in[ARRAY_SIZE]; //h_开头的变量 表示 插在cpu上的变量    host  CPU编程 
  printf("===========\n"); 
  // ARRAY_SIZE=8  8个元素的数组  
  for(int i=0;i<ARRAY_SIZE;i++){  //给h_in[ARRAY_SIZE]数组的元素赋值
    printf("%d \n", i );

    //h_in[i] = float(i); // 类型转换
    h_in[i] = i; // 类型转换
    printf("============================\n"); 
    printf("h_in = %f \n", h_in[i]);
  }
  float h_out[ARRAY_SIZE]; 

  // declare GPU memory pointers
  float* d_in; //d_  表示GPU上的变量 device  GPU编程  指向GPU的地方 指针类型 指向float变量
  float* d_out; // 

  //要把 h_in数组 复制给d_in  
  // 最后是 d_out  复制给 h_out
  // allocate GPU memory  给GPU分配内存空间  
  cudaMalloc((void**) &d_in,ARRAY_BYTES);   // 给d_in  和 d_out都分配了内存空间，空间大小是ARRAY_BYTES
  cudaMalloc((void**) &d_out,ARRAY_BYTES);  //  d_out 传进去的是d_out的地址 ，  要修改d_out本身的值， 而不是d_out指向的值 ， 也不是指向d_out的地址
//把d_out看成普通变量， &d_out相当于是传递d_out的地址，而d_out是地址指针变量， 
//一个指针类型， 该类型的指针指向无类型指针
//https://zhuanlan.zhihu.com/p/88652961


  // transfer the array to GPU  第二步  cuda 内存复制 ，  
  cudaMemcpy(d_in,h_in,ARRAY_BYTES,cudaMemcpyHostToDevice); 
  // 复制的目标位置， 复制原位置 从host to device  复制量 

  // launch the kernel
  square<<<1,ARRAY_SIZE>>>(d_out,d_in);  //有1个线程块  有8个线程， 每个线程一起算
  //1 加载在多少线程块  创建爱你多少线程块  ARRAY_SIZE 每个线程块 多少线程
  //square的输入参数  __global__ void square(float* d_out,float* d_in)

  // copy back the result array to the GPU
  cudaMemcpy(h_out,d_out,ARRAY_BYTES,cudaMemcpyDeviceToHost); 
  // 目标host  原位置d  从device复制到host   把d-out从GPU复制到CPU的h-out中

  // print out the resulting array
  for(int i=0;i<ARRAY_SIZE;i++){
    printf("===============\n");
    // printf("%f",h_out[i]);
    printf("%f \n",h_out[i]);
    //printf(((i%4) != 3) ? "\t" : "\n");
  }

  // free GPU memory allocation
  cudaFree(d_in); //释放GPU 和 CPU的
  cudaFree(d_out);

  return 0;


}
