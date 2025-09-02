#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define M 1024 
#define N 512 
#define P 2048 
#define TILE_WIDTH 32

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}


void init_matrix(float *mat , int row , int col){
    for(int i=0;i<row * col ;i++){
        mat[i]=(float)rand()/RAND_MAX;
    }
}


int main()
{
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

    // Kernel execution
    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dim_grid(ceil(P/(float)(TILE_WIDTH)), ceil(M/(float)(TILE_WIDTH)), 1);
    tiled_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, P);

    // Copy back results
    cudaError_t err_C_ = cudaMemcpy(C, d_C, M*P*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err_C_);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}