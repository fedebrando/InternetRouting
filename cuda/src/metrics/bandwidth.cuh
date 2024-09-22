
/*
 * Widest Path
*/

#ifndef BANDWIDTH 
#define BANDWIDTH

#include <stdio.h>
#include <math.h>
#include "utilities.cuh"

#define BANDWIDTH_ZERO 0
#define BANDWIDTH_UNITY INFINITY

__host__ __device__ double bandwidth_plus(double bw1, double bw2);
__host__ __device__ double bandwidth_times(double bw1, double bw2);
__host__ __device__ boolean bandwidth_less_eq(double bw1, double bw2);
__host__ __device__ boolean bandwidth_less(double bw1, double bw2);
__host__ __device__ void bandwidth_print(double bw);

#endif
