#include <bits/stdc++.h>
#include <cuda_runtime.h>


#define block_size 16
#define N 32



//here we are using skewed row to avoid bank conflicts

__global__ void skewedRowTranspose(float *out, float *in,int width){
    __shared__ float tile[block_size][block_size];
    int x=blockIdx.x*block_size + threadIdx.x;
    int y=blockIdx.y*block_size + threadIdx.y;

    if(x<width && y<width){
      tile[threadIdx.y][(threadIdx.x+threadIdx.y)%block_size]=in[y*width + x];
    }

    __syncthreads();

    int transposed_x=blockIdx.y*block_size + threadIdx.x;
    int transposed_y=blockIdx.x*block_size + threadIdx.y;

    if(transposed_x<width && transposed_y<width){
      out[transposed_y*width + transposed_x]=tile[(threadIdx.x+threadIdx.y)%block_size][threadIdx.y];
    }
}

int main(){

  const int size=N*N;
  float *h_in=(float *)malloc(size*sizeof(float));
  float *h_out=(float *)malloc(size*sizeof(float));

  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      h_in[i*N + j]=i*N + j;
    }
  }
  float *d_in,*d_out;
  cudaMalloc((void **)&d_in,size*sizeof(float));
  cudaMalloc((void **)&d_out,size*sizeof(float));
  cudaMemcpy(d_in,h_in,size*sizeof(float),cudaMemcpyHostToDevice);
  dim3 block(block_size,block_size);
  dim3 grid((N + block.x - 1)/block.x,(N + block.y - 1)/block.y);
  skewedRowTranspose<<<grid,block>>>(d_out,d_in,N);
  cudaMemcpy(h_out,d_out,size*sizeof(float),cudaMemcpyDeviceToHost);
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      printf("%f ",h_out[i*N + j]);
    }
    printf("\n");
  }
  free(h_in);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
