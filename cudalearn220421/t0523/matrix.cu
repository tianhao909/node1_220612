#include "/usr/local/cuda-11.3/targets/x86_64-linux/include/cuda_runtime.h"
#include<stdlib.h>
#include <stdio.h>
#include <assert.h>

#define N 64

__global__ void gpu(int *a, int *b, int *c_gpu){
    int r = blockDim.x * blockIdx.x + threadIdx.x; //blockDim.x 块的线程数， blockIdx.x 当前块的索引  threadIdx.x当前块中线程的索引 
    int c = blockDim.y * blockIdx.y + threadIdx.y;

    if(r < N &&  c<N){
        c_gpu[r*N +c] = a[r*N + c] + b[r*N + c];
    }
}

void cpu(int *a, int *b, int *c_cpu){

    for(int r=0; r<N ; r++){
        for(int c=0; c<N; c++){
            c_cpu[r*N +c] = a[r*N + c] + b[r*N + c];
        }
    }
}


bool check(int *a, int *b, int * c_gpu, int *c_cpu){

    for(int r=0; r<N ; r++){
        for(int c=0; c<N; c++){
            if ( c_cpu[r*N +c] !=  c_gpu[r*N +c] ){
                return false;
            } else {
                return true;
            }
        }
    }
    return true;
}

 

int main() {
    int *a, *b, *c_cpu, *c_gpu;
    size_t size = N * N * sizeof(int);

    //统一内存是可从系统中的任何处理器访问的单个内存地址空间（参见上图）。 这种硬件/软件技术允许应用程序分配可以从 CPU 或 GPU 上运行的代码读取或写入的数据。 分配统一内存就像用调用 cudaMallocManaged() 替换对 malloc() 或 new 的调用一样简单，这是一个分配函数，它返回一个可从任何处理器访问的指针（下文中的 ptr）。
    //cudaError_t cudaMallocManaged(void** ptr, size_t size);
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c_cpu, size);
    cudaMallocManaged(&c_gpu, size);
    
    for(int r=0; r<N; r++){
        for(int c=0; c<N ; c++){
            a[r*N + c] = r;
            b[r*N + c] = c;
            c_gpu[r*N + c] = 0;
            c_cpu[r*N + c] = 0;
        }
    }
    
    dim3 threads(16,16,1); //此类型是基于uint3的整数向量类型，用于指定维度。定义dim3类型的变量时，保留为未指定的任何组件将初始化为1。
    dim3 blocks((N+threads.x -1)/ threads.x, (N+threads.y-1) / threads.y, 1); //上取整操作

    gpu<<<blocks, threads>>>(a,b,c_gpu);//通过在函数名称和圆括号括起的参数列表之间插入<<< Dg, Db, Ns, S >>>形式的表达式，来定义执行配置，  
    //  Dg是类型dim3（参见4.3.1.2），用于指定网格的维度和大小，因此Dg.x*Dg.y等于要启动的块数；Dg.z未使用；
    //  Db是类型dim3（参见4.3.1.2），用于指定每块的维度和大小，因此Dg.x*Dg.y*Db.z等于每块的线程数；
    //  Ns是类型size_t，用于指定为此调用按块动态分配的共享内存中的字节数以及静态分配的内存；此动态分配的内存由声明为外部数组的任何一个变量使用，如4.2.2.3所述；Ns是默认值为0的可选参数；
    //  S是类型cudaStream_t，用于指定相关联的流；S是默认值为0的可选参数。

    cudaDeviceSynchronize();
    
    cpu(a,b,c_cpu);
    check(a,b,c_cpu,c_gpu) ? printf("ok") : printf("error");

    cudaFree(a); 
    cudaFree(b);
    cudaFree(c_cpu);
    cudaFree(c_gpu);



}