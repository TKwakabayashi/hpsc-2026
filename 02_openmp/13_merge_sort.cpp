#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

void merge(std::vector<int>& vec, int begin, int mid, int end) {
  std::vector<int> tmp(end-begin+1);
  int left = begin;
  int right = mid+1;
  for (int i=0; i<tmp.size(); i++) { 
    if (left > mid)
      tmp[i] = vec[right++];
    else if (right > end)
      tmp[i] = vec[left++];
    else if (vec[left] <= vec[right])
      tmp[i] = vec[left++];
    else
      tmp[i] = vec[right++]; 
  }
  for (int i=0; i<tmp.size(); i++) 
    vec[begin++] = tmp[i];
}

void merge_sort(std::vector<int>& vec, int begin, int end, int depth) {
  if(begin < end) {
    int mid = (begin + end) / 2;
    if (depth > 0) {
      #pragma omp task
      merge_sort(vec, begin, mid, depth-1);
      #pragma omp task
      merge_sort(vec, mid+1, end, depth-1);
      #pragma omp taskwait
      merge(vec, begin, mid, end);
    } else {
      merge_sort(vec, begin, mid, 0);
      merge_sort(vec, mid+1, end, 0);
      merge(vec, begin, mid, end);
    }
  }
}

int main() {
  int n = 20;
  std::vector<int> vec(n);
  for (int i=0; i<n; i++) {
    vec[i] = rand() % (10 * n);
    printf("%d ",vec[i]);
  }
  printf("\n");

  int max_thread = omp_get_max_threads();
  int max_depth = 0;
  while ((1 << max_depth) < max_thread) max_depth++;
  #pragma omp parallel
  {
    #pragma omp single
    merge_sort(vec, 0, n-1, max_depth);
  }

  for (int i=0; i<n; i++) {
    printf("%d ",vec[i]);
  }
  printf("\n");
}
