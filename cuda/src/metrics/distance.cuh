
/*
 * Shortest Path
*/

#ifndef DISTANCE
#define DISTANCE

#include <math.h>
#include "utilities.cuh"

#define DISTANCE_ZERO INFINITY
#define DISTANCE_UNITY 0

__host__ __device__ double distance_plus(double d1, double d2);
__host__ __device__ double distance_times(double d1, double d2);
__host__ __device__ boolean distance_less_eq(double d1, double d2);
__host__ __device__ boolean distance_less(double d1, double d2);

#endif
