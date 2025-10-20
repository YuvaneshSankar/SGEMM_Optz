#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;


#define BLOCK_SIZE 16
#define N 32


#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(-1); \
    }


__global__ void skewedRowTranspose(float *out, const float *in, int width) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < width && y < width) {
        tile[threadIdx.y][(threadIdx.x + threadIdx.y) % BLOCK_SIZE] = in[y * width + x];
    }

    __syncthreads();

    int x_t = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int y_t = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    if (x_t < width && y_t < width) {
        out[y_t * width + x_t] = tile[(threadIdx.x + threadIdx.y) % BLOCK_SIZE][threadIdx.y];
    }
}


int main() {
    const int size = N * N;
    const size_t bytes = size * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            h_in[i * N + j] = i * N + j;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);


    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    skewedRowTranspose<<<grid, block>>>(d_out, d_in, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));


    printf("\nGPU Skewed Transpose Completed in %.3f ms\n", ms);
    printf("Sample Output Matrix (%d x %d):\n", N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            printf("%5.0f ", h_out[i * N + j]);
        printf("\n");
    }

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
