#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

   __m512i jvec = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  __m512 xj = _mm512_load_ps(x);
  __m512 yj = _mm512_load_ps(y);
  __m512 mj = _mm512_load_ps(m);

  for(int i=0; i<N; i++) {
    __m512i ivec = _mm512_set1_epi32(i);
    __mmask16 mask = _mm512_cmp_epi32_mask(ivec, jvec, _MM_CMPINT_NE);

    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);

    __m512 rx = _mm512_sub_ps(xi, xj);
    __m512 ry = _mm512_sub_ps(yi, yj);

    __m512 invr = _mm512_rsqrt14_ps(_mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry)));
    __m512 invr3 = _mm512_mul_ps(invr, _mm512_mul_ps(invr, invr));

    __m512 dfx = _mm512_mul_ps(rx,  _mm512_mul_ps(mj, invr3));
    __m512 dfy = _mm512_mul_ps(ry,  _mm512_mul_ps(mj, invr3));

    dfx = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), dfx);
    dfy = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), dfy);
      

    fx[i] -= _mm512_reduce_add_ps(dfx);
    fy[i] -= _mm512_reduce_add_ps(dfy);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
