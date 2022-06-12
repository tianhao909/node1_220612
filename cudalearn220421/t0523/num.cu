#include "/usr/local/cuda-11.3/targets/x86_64-linux/include/cuda_runtime.h"
#include<stdlib.h>
#include <stdio.h>
#include <assert.h>

#define N 10

__global__ void gpu(int num) {
    printf("%d\n",num);
}

int main(){
    for(int i=0; i<N; i++){
        //gpu<<<1,2>>>(i);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        gpu<<<1,1,0,stream>>>(i);
        cudaStreamDestroy(stream);
    }
    cudaDeviceSynchronize();

}
