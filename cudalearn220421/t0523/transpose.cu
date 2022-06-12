#include "/usr/local/cuda-11.3/targets/x86_64-linux/include/cuda_runtime.h"
#include<stdlib.h>
#include <stdio.h>
#include <assert.h>

#define TILE_DIM 32
#define BLOCK_SIZE 8
#define MX 20480
#define MY 20480  

__global__ void transpose(float* odata, float* idata){
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    int w = gridDim.x * TILE_DIM;
    if(x >= MX || y >= MY){
        return;
    }
    for(int i=0; i < TILE_DIM; i += BLOCK_SIZE){
        odata[x*w + y + i] = idata[(y+i) * w + x];
    }
}

bool check(float *c_cpu, float* c_gpu){
    for(int r = 0; r < MX; r++){
        for(int c= 0; c < MY; c++){
            if(c_cpu[r * MX + c] != c_gpu[r * MY + c]){
                return false;
            }
        }
    }
    return true;
}

int main(){
    size_t size = MX * MY * sizeof(float);
    float *h_idata, *h_odata, *d_idata, *d_odata, *res;
    cudaMallocHost(&h_idata, size);
    cudaMallocHost(&h_odata, size);
    cudaMalloc(&d_idata, size);
    cudaMalloc(&d_odata, size);
    dim3 threads(TILE_DIM, BLOCK_SIZE, 1);
    dim3 blocks( (MX + TILE_DIM - 1) / TILE_DIM, (MY + TILE_DIM - 1) / TILE_DIM, 1 );
    for(int i=0; i<MX; i++){
        for(int j=0; j<MY; j++){  
            h_idata[i * MY + j] = i * MY + j;
            res[i * MY + j] = j * MY  + i;
        }
    }
    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);
    transpose<<<blocks, threads>>>(d_odata, d_idata);
    cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
    check(res, h_odata) ? printf("ok") : printf("error");
    cudaFreeHost(h_idata);
    cudaFreeHost(h_odata);
    cudaFreeHost(res);
    cudaFree(d_idata);
    cudaFree(d_odata);
}


