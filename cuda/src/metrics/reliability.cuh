
/*
 * Most Reliable Path
*/

#ifndef RELIABILITY 
#define RELIABILITY

#include "utilities.cuh"

#define RELIABILITY_ZERO 0
#define RELIABILITY_UNITY 1

__host__ __device__ double reliability_plus(double r1, double r2);
__host__ __device__ double reliability_times(double r1, double r2);
__host__ __device__ boolean reliability_less_eq(double r1, double r2);
__host__ __device__ boolean reliability_less(double r1, double r2);

#endif
