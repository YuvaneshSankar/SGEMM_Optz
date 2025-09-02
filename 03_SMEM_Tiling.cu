#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define M 1024 
#define N 512 
#define P 2048 
#define TILE_WIDTH 32

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}



__global__ void tiled_mat_mul_kernel(float *A, float *B, float *C, int m, int n, int p){

    // Ensure that TILE_WIDTH = BLOCK_SIZE
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);

    int by=blockIdx.y;
    int bx=blockIdx.x;

    int ty=threadIdx.y;
    int tx=threadIdx.x;

    int i=by*TILE_WIDTH + ty;
    int j=bx*TILE_WIDTH + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float val=0.0f;

    for(int tileId=0;tileId<ceil((float)n/TILE_WIDTH);tileId++){

        if(i<m && (tileId*TILE_WIDTH + tx)<n){
            sh_A[ty][tx]=A[i*n + tileId*TILE_WIDTH + tx];
        }
        else{
            sh_A[ty][tx]=0.0f;
        }
        if(j<p && (tileId*TILE_WIDTH + ty)<n){
            sh_B[ty][tx]=B[(tileId*TILE_WIDTH + ty)*p + j];
        }
        else{
            sh_B[ty][tx]=0.0f;
        }
        __syncthreads();


        for(int k=0;k<TILE_WIDTH;k++){
            val+=sh_A[ty][k]*sh_B[k][tx];
        }
        __syncthreads();

    }

    if(i<m && j<p){
        C[i*p + j]=val;
    }
}



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

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}