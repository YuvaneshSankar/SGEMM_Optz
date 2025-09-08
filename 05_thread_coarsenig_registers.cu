#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define N 1024
#define a 2.0f
#define b 1.0f
__global__ void kernel(float *H,int n){
    int x= blockIdx.x* blockDim.x + threadIdx.x;
    int stride=4;
    int till=x*stride;

    float A=a;
    float B=b;

    if(x<n){
        for(int i=0;i<stride;i++){
            int idx=till+i;
            if(idx<n){
                H[idx]=A*H[idx]+B;
            }
        }
    }
}

int main(){
    float *H;
    float *d_H;
    H=(float*)malloc(N*sizeof(float));
    for(int i=0;i<N;i++){
        H[i]=i;
    }

    cudaMalloc((void**)&d_H,N*sizeof(float));
    cudaMemcpy(d_H,H,N*sizeof(float),cudaMemcpyHostToDevice);

    kernel<<<(N/4+255)/256,256>>>(d_H,N);

    cudaMemcpy(H,d_H,N*sizeof(float),cudaMemcpyDeviceToHost);

    for(int i=0;i<10;i++){
        std::cout<<H[i]<<" ";
    }
    std::cout<<std::endl;

    cudaFree(d_H);
    free(H);
    return 0;
}