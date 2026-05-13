#include <cstdio>
#include <cstdlib>

__global__ void initBucket(int* bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) bucket[i] = 0;
}

__global__ void countKeys(int* key, int* bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&bucket[key[i]], 1);
}

__global__ void prefixSum(int* bucket, int* prefix, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) {
    int sum = 0;
    for (int k = 0; k < i; k++)
      sum += bucket[k];
    prefix[i] = sum;
  }
}

__global__ void sortKeys(int* key, int* bucket, int* prefix, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) {
    int start = prefix[i];
    int count = bucket[i];
    for (int j = 0; j < count; j++)
      key[start + j] = i;
  }
}

int main() {
  int n = 50;
  int range = 5;
  const int M = 1024;

  int *key, *bucket, *prefix;
  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));
  cudaMallocManaged(&prefix, range * sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  initBucket<<<(range + M - 1) / M, M>>>(bucket, range);
  countKeys<<<(n + M - 1) / M, M>>>(key, bucket, n);
  prefixSum<<<(range + M - 1) / M, M>>>(bucket, prefix, range);
  sortKeys<<<(range + M - 1) / M, M>>>(key, bucket, prefix, range);
  cudaDeviceSynchronize();


  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
  cudaFree(prefix);
}
