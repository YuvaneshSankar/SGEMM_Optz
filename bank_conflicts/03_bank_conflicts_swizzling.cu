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

__global__ void swizzling(float* in, float* out, int width)
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < width && y < width)
        tile[threadIdx.y][threadIdx.x ^ threadIdx.y] = in[y * width + x];

    __syncthreads();

    int tx = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int ty = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    if (tx < width && ty < width)
        out[ty * width + tx] = tile[threadIdx.x ^ threadIdx.y][threadIdx.y];
}


void cpuTranspose(float* out, const float* in, int width)
{
    for (int i = 0; i < width; i++)
        for (int j = 0; j < width; j++)
            out[j * width + i] = in[i * width + j];
}


int main()
{
    const int size = N * N;
    const size_t bytes = size * sizeof(float);


    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            h_in[i * N + j] = i * N + j;


    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));


    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    swizzling<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));


    cpuTranspose(h_ref, h_in, N);

    bool correct = true;
    for (int i = 0; i < size; i++) {
        if (fabs(h_ref[i] - h_out[i]) > 1e-5) {
            correct = false;
            printf("Mismatch at index %d: CPU=%f GPU=%f\n", i, h_ref[i], h_out[i]);
            break;
        }
    }

    printf("\nCUDA Swizzling Transpose completed in %.4f ms\n", ms);
    printf("Matrix transpose correct: %s\n", correct ? "YES" : "NO");

    printf("\nSample output matrix (%dx%d):\n", N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            printf("%5.0f ", h_out[i * N + j]);
        printf("\n");
    }


    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    free(h_ref);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
