#include <bits/stdc++.h>
#include <cuda_runtime.h>


#define N 1024

__global__ void kernel(float *A,int n){
    int x= blockIdx.x* blockDim.x + threadIdx.x;
    int stride=4;
    int till=x*stride;

    if(x<n){
        //here one thread does 4 operations which is redudant 
        for(int i=0;i<stride;i++){
            int idx=till+i;
            if(idx<n){
                A[idx]+=1.0f;
            }
        }
    }
}

int main(){
    float *h_A,*d_A;
    int size_A=N*sizeof(float);
    h_A=(float*)malloc(size_A);
    cudaMalloc(&d_A,size_A);
    for(int i=0;i<N;i++){
        h_A[i]=1.0f;
    }
    cudaMemcpy(d_A,h_A,size_A,cudaMemcpyHostToDevice);
    //toatl threads is 1024 but as we are coarsening by 4 so we need 1024/4 threads -> 256 threads
    //hence we are launching 256 threads
    kernel<<<(N/4 + 255)/256, 256>>>(d_A, N);
    cudaMemcpy(h_A,d_A,size_A,cudaMemcpyDeviceToHost);
    for(int i=0;i<10;i++){
        std::cout<<h_A[i]<<" ";
    }
    std::cout<<std::endl;
    cudaFree(d_A);
    free(h_A);
    return 0;
}