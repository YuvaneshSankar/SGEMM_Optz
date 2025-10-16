  #include <cuda_runtime.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <time.h>

  // Dimensions
  #define N 32  // 32x32 matrix (warp size)
  #define BLOCK_SIZE N

  // Kernel WITHOUT padding - causes bank conflicts in shared memory column access
  __global__ void transposeNoPadding(float *out, const float *in, int width) {
      __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];  // 32x32 - no padding

      int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
      int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

      // Load input into shared memory (row-wise access - no conflict)
      if (x < width && y < width) {
          tile[threadIdx.y][threadIdx.x] = in[y * width + x];
      }
      __syncthreads();

      // Transpose: Access column-wise (tile[threadIdx.x][threadIdx.y]) - BANK CONFLICTS HERE!
      // For first column access example: tile[threadIdx.x][0] would all hit bank 0
      int x_t = blockIdx.y * BLOCK_SIZE + threadIdx.x;
      int y_t = blockIdx.x * BLOCK_SIZE + threadIdx.y;
      if (x_t < width && y_t < width) {
          out[y_t * width + x_t] = tile[threadIdx.x][threadIdx.y];  // Column access causes conflict
      }
  }

  // Kernel WITH padding - avoids bank conflicts
  __global__ void transposeWithPadding(float *out, const float *in, int width) {
      __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];  // 32x33 - +1 padding column (unused)

      int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
      int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

      // Load input into shared memory (row-wise access - no conflict)
      if (x < width && y < width) {
          tile[threadIdx.y][threadIdx.x] = in[y * width + x];
      }
      __syncthreads();

      // Transpose: Access column-wise (tile[threadIdx.x][threadIdx.y]) - NO CONFLICTS with padding!
      // For first column access: tile[threadIdx.x][0] now hits different banks
      int x_t = blockIdx.y * BLOCK_SIZE + threadIdx.x;
      int y_t = blockIdx.x * BLOCK_SIZE + threadIdx.y;
      if (x_t < width && y_t < width) {
          out[y_t * width + x_t] = tile[threadIdx.x][threadIdx.y];  // Column access distributed across banks
      }
  }

  int main() {
      const int size = N * N;
      float *h_in = (float*)malloc(size * sizeof(float));
      float *h_out_no_pad = (float*)malloc(size * sizeof(float));
      float *h_out_with_pad = (float*)malloc(size * sizeof(float));

      // Initialize input matrix
      srand(time(NULL));
      for (int i = 0; i < size; i++) {
          h_in[i] = rand() / (float)RAND_MAX;
      }

      // Allocate GPU memory
      float *d_in, *d_out_no_pad, *d_out_with_pad;
      cudaMalloc(&d_in, size * sizeof(float));
      cudaMalloc(&d_out_no_pad, size * sizeof(float));
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
      cudaEventRecord(start);
      transposeNoPadding<<<grid, block>>>(d_out_no_pad, d_in, N);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float time_no_pad;
      cudaEventElapsedTime(&time_no_pad, start, stop);

      // Run with padding
      cudaEventRecord(start);
      transposeWithPadding<<<grid, block>>>(d_out_with_pad, d_in, N);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float time_with_pad;
      cudaEventElapsedTime(&time_with_pad, start, stop);

      // Copy results back
      cudaMemcpy(h_out_no_pad, d_out_no_pad, size * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_out_with_pad, d_out_with_pad, size * sizeof(float), cudaMemcpyDeviceToHost);

      // Verify correctness (both should produce same transpose)
      bool correct = true;
      for (int i = 0; i < size; i++) {
          if (fabs(h_out_no_pad[i] - h_out_with_pad[i]) > 1e-5) {
              correct = false;
              break;
          }
      }

      // Output
      printf("Matrix transpose completed.\n");
      printf("No padding time: %.3f ms (expected conflicts)\n", time_no_pad);
      printf("With padding time: %.3f ms (no conflicts)\n", time_with_pad);
      printf("Results match: %s\n", correct ? "Yes" : "No");

      // Cleanup
      free(h_in);
      free(h_out_no_pad);
      free(h_out_with_pad);
      cudaFree(d_in);
      cudaFree(d_out_no_pad);
      cudaFree(d_out_with_pad);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      return 0;
  }
