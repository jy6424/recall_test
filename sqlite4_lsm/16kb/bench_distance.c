/*
** Standalone micro-benchmark for vectorF8DistanceCos.
** Tests whether the distance function itself is the bottleneck.
**
** Build (sqlite4, same flags as library):
**   gcc -g -O2 -fPIC -DNDEBUG=1 bench_distance.c -lm -o bench_dist
**
** Build (sqlite3-equivalent flags):
**   gcc -g -O0 bench_distance.c -lm -o bench_dist_O0
**
** Run: ./bench_dist
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* --- Inlined from vectorfloat8.c --- */

typedef unsigned char u8;
typedef unsigned int u32;
typedef uint16_t u16;
typedef uint64_t u64;

#define ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

typedef u16 VectorType;
typedef u16 VectorDims;

struct Vector {
  VectorType type;
  u16 flags;
  VectorDims dims;
  void *data;
};
typedef struct Vector Vector;

static void deserializeF32_(const u8 *p, float *out) {
  u32 v;
  memcpy(&v, p, sizeof(v));
  memcpy(out, &v, sizeof(float));
}

static void vectorF8GetParameters(const u8 *pData, int dims, float *pAlpha, float *pShift){
  pData = pData + ALIGN(dims, sizeof(float));
  deserializeF32_(pData, pAlpha);
  deserializeF32_(pData + sizeof(float), pShift);
}

static float vectorF8DistanceCos(const Vector *v1, const Vector *v2){
  int i;
  float alpha1, shift1, alpha2, shift2;
  u32 sum1 = 0, sum2 = 0, sumsq1 = 0, sumsq2 = 0, doti = 0;
  float dot = 0, norm1 = 0, norm2 = 0;
  u8 *data1 = v1->data, *data2 = v2->data;

  vectorF8GetParameters(v1->data, v1->dims, &alpha1, &shift1);
  vectorF8GetParameters(v2->data, v2->dims, &alpha2, &shift2);

  for(i = 0; i < v1->dims; i++){
    sum1 += data1[i];
    sum2 += data2[i];
    sumsq1 += data1[i]*data1[i];
    sumsq2 += data2[i]*data2[i];
    doti += data1[i]*data2[i];
  }

  dot = alpha1 * alpha2 * (float)doti + alpha1 * shift2 * (float)sum1 + alpha2 * shift1 * (float)sum2 + shift1 * shift2 * v1->dims;
  norm1 = alpha1 * alpha1 * (float)sumsq1 + 2 * alpha1 * shift1 * (float)sum1 + shift1 * shift1 * v1->dims;
  norm2 = alpha2 * alpha2 * (float)sumsq2 + 2 * alpha2 * shift2 * (float)sum2 + shift2 * shift2 * v1->dims;

  return 1.0 - (dot / sqrt(norm1 * norm2));
}

/* --- Benchmark --- */

#define DIMS 1024
#define NUM_VECTORS 100   /* simulate ~100 edge vectors per node */
#define NUM_ITERS 200000  /* ~20M distance computations total */

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Allocate a float8 vector with random data + alpha/shift parameters */
static u8 *alloc_f8_vector(int dims) {
  int dataSize = ALIGN(dims, sizeof(float)) + 2 * sizeof(float);
  u8 *buf = (u8*)malloc(dataSize);
  int i;
  float alpha = 0.01f, shift = -0.5f;

  for (i = 0; i < dims; i++) {
    buf[i] = (u8)(rand() % 256);
  }
  /* Pad between dims and parameters */
  for (i = dims; i < ALIGN(dims, sizeof(float)); i++) {
    buf[i] = 0;
  }
  /* Write alpha and shift */
  memcpy(buf + ALIGN(dims, sizeof(float)), &alpha, sizeof(float));
  memcpy(buf + ALIGN(dims, sizeof(float)) + sizeof(float), &shift, sizeof(float));

  return buf;
}

int main(void) {
  int i, j;
  Vector query;
  Vector edges[NUM_VECTORS];
  float total = 0;
  double t0, t1, elapsed;

  srand(42);

  /* Allocate query vector */
  query.type = 6;  /* VECTOR_TYPE_FLOAT8 */
  query.flags = 0;
  query.dims = DIMS;
  query.data = alloc_f8_vector(DIMS);

  /* Allocate edge vectors */
  for (i = 0; i < NUM_VECTORS; i++) {
    edges[i].type = 6;
    edges[i].flags = 0;
    edges[i].dims = DIMS;
    edges[i].data = alloc_f8_vector(DIMS);
  }

  printf("Benchmark: %d distance computations (%d iters x %d vectors), dims=%d\n",
         NUM_ITERS * NUM_VECTORS, NUM_ITERS, NUM_VECTORS, DIMS);

  t0 = now_ms();

  for (i = 0; i < NUM_ITERS; i++) {
    for (j = 0; j < NUM_VECTORS; j++) {
      total += vectorF8DistanceCos(&query, &edges[j]);
    }
  }

  t1 = now_ms();
  elapsed = t1 - t0;

  printf("Total time: %.1f ms\n", elapsed);
  printf("Per distance: %.3f us\n", elapsed * 1000.0 / (NUM_ITERS * NUM_VECTORS));
  printf("Distances/sec: %.0f\n", (double)(NUM_ITERS * NUM_VECTORS) / (elapsed / 1000.0));
  printf("(checksum: %f)\n", total);

  /* Cleanup */
  free(query.data);
  for (i = 0; i < NUM_VECTORS; i++) {
    free(edges[i].data);
  }

  return 0;
}
