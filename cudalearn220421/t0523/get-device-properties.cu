#include "/usr/local/cuda-11.3/targets/x86_64-linux/include/cuda_runtime.h"
#include<stdlib.h>
#include <stdio.h>
#include <assert.h>
int main(){
    int id;
    cudaGetDevice(&id);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, id);

    printf("device id: %d\n sms: %d\n capability major:%d\n capability minor:%d\n warp size: %d\n", id, props.multiProcessorCount, props.major, props.minor, props.warpSize);


}