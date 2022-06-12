// 导入cuda相关模块
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
// 声明一个函数后面需要调用
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//它是调用GPU运算的入口函数，返回类型是cudaError_t。
__global__ void addKernel(int *c, const int *a, const int *b)
{
    // 这个函数是运行在GPU上的，称为核函数
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    // cuda_Error_t指的是cuda错误类型，取值为整数
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    // cudaDeviceProp是设备属性结构体
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1; 
        //return 0;表示正常结束
        //return非0表示非正常结束
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called to Reset and Synchronize before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    // 
    cudaStatus = cudaDeviceReset(); 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
// 用于使用CUDA并行添加向量的辅助函数
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    // GPU设备端的数据指针
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    // GPU状态指示
    cudaError_t cudaStatus;
    printf("cudaStatus:%s\n",cudaStatus);

    
    // Choose which GPU to run on, change this on a multi-GPU system.
    // 当电脑上有多个显卡支持CUDA时，选择某个GPU来进行运算。0表示设备号，还有1,2,3……
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    // 分配GPU设备端的内存。三个变量，所以需要分配三次。
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    // 申请失败就报错
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    // 拷贝数据到GPU中，已有两个变量(a,b)，所以需要拷贝两次。c是空的，不管。
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    // 拷贝数据到GPU中，已有两个变量(a,b)，所以需要拷贝两次。c是空的，不管。
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    // 一个grid,size个线程，每个线程的函数是addKernel，参数是dev_c, dev_a, dev_b
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    // 同步线程（cuda为了防止混乱直接不打印）
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    // 将结果拷贝回主机，不然怎么在主机上显示
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    // 释放GPU设备端内存
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
