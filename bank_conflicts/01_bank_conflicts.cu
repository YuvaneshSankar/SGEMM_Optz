  #include <cuda_runtime.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <time.h>

  // Dimensions
  #define N 32  // 32x32 matrix (warp size)
  #define BLOCK_SIZE 16


  __global__ void transposeWithPadding(float *out , float *in , int width){
        __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1]; // +1 to avoid bank conflicts
        int x=blockIdx.x*BLOCK_SIZE + threadIdx.x;
        int y=blockIdx.y*BLOCK_SIZE + threadIdx.y;
        if(x<width && y<width){
            tile[threadIdx.y][threadIdx.x]=in[y*width + x];
        }
        __syncthreads();

        //now lets transpose the matrix okay
        //here we do block swapping so we swap blockIdx.x and blockIdx.y
        x=blockIdx.y*BLOCK_SIZE + threadIdx.x; //note the swap of blockIdx.x and blockIdx.y
        y=blockIdx.x*BLOCK_SIZE + threadIdx.y;
        if(x<width && y<width){
            out[y*width + x]=tile[threadIdx.x][threadIdx.y]; //normal transpose by changing indices of x and y
        }

  }


  int main() {
      const int size = N * N;
      float *h_in = (float*)malloc(size * sizeof(float));
      float *h_out_with_pad = (float*)malloc(size * sizeof(float));

      // Initialize input matrix
      srand(time(NULL));
      for (int i = 0; i < size; i++) {
          h_in[i] = rand() / (float)RAND_MAX;
      }

      // Allocate GPU memory
      float *d_in, *d_out_with_pad;
      cudaMalloc(&d_in, size * sizeof(float));
      cudaMalloc(&d_out_with_pad, size * sizeof(float));

      // Copy input to GPU
      cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);

      // Grid dimensions (for 32x32 matrix)
      dim3 block(BLOCK_SIZE, BLOCK_SIZE);
      dim3 grid(1, 1);  // Single block for simplicity

      // Run without padding
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);


      // Run with padding
      cudaEventRecord(start);
      transposeWithPadding<<<grid, block>>>(d_out_with_pad, d_in, N);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float time_with_pad;
      cudaEventElapsedTime(&time_with_pad, start, stop);

      // Copy results back
      cudaMemcpy(h_out_with_pad, d_out_with_pad, size * sizeof(float), cudaMemcpyDeviceToHost);

      printf("With padding time: %.3f ms (no conflicts)\n", time_with_pad);

      // Cleanup
      free(h_in);
      free(h_out_with_pad);
      cudaFree(d_in);
      cudaFree(d_out_with_pad);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      return 0;
  }
