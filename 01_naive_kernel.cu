#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <time.h>



#define M 1024 
#define N 512 
#define P 2048  
#define BLOCK_SIZE 32


__global__ void naive_kernel(float *A, float *B, float *C, int m, int n, int p){
    const int x= blockIdx.x * blockDim.x + threadIdx.x;
    const int y= blockIdx.y * blockDim.y + threadIdx.y;

    if(x<m && y<p){
        float sum=0.0f;
        for(int k=0;k<n;k++){
            sum+=A[x*n +k] * B[k*p + y];
        }
        C[x*p+y]=sum;
    }
}



void init_matrix(float *mat , int row , int col){
    for(int i=0;i<row * col ;i++){
        mat[i]=(float)rand()/RAND_MAX;
    }
}




int main(){
    float *h_A,*h_B,*h_C;
    float *d_A,*d_B,*d_C;

    int size_A=M*N*sizeof(float);
    int size_B=N*P*sizeof(float);
    int size_C=M*P*sizeof(float);

    //allocate host memeory
    h_A=(float*)malloc(size_A);
    h_B=(float*)malloc(size_B);
    h_C=(float*)malloc(size_C);

    //intialize matrices
    init_matrix(h_A,M,N);
    init_matrix(h_B,N,P);

    //allocate device memeory
    cudaMalloc(&d_A,size_A);
    cudaMalloc(&d_B,size_B);
    cudaMalloc(&d_C,size_C);

    //copt data from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    //launch kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((P + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    //warm up runs

    for(int i=0;i<3;i++){
        naive_kernel<<<gridDim,blockDim>>>(d_A,d_B,d_C,M,N,P);
        cudaDeviceSynchronize();
    }


    //Benchmark implementation
    double gpu_total_time=0.0;
    for(int i=0;i<20;i++){
        naive_kernel<<<gridDim,blockDim>>>(d_A,d_B,d_C,M,N,P);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;   
    
}