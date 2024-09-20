
/*
 * Widest path
*/

#ifndef BANDWIDTH 
#define BANDWIDTH

#include <math.h>

#define BANDWIDTH_ZERO 0
#define BANDWIDTH_UNITY INFINITY

__host__ __device__ double bandwidth_plus(double bw1, double bw2);
__host__ __device__ double bandwidth_times(double bw1, double bw2);
__host__ __device__ int bandwidth_less_eq(double bw1, double bw2);
__host__ __device__ int bandwidth_less(double bw1, double bw2);

#endif
