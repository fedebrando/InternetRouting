
/*
 * Widest Shortest Path or Most Reliable Shortest Path
*/

#ifndef WSP_OR_MRSP
#define WSP_OR_MRSP

#include <stdio.h>
#include "utilities.cuh"
#include "lex_product.cuh"
#include "distance.cuh"
#include "metric_hyperparams.h"

#ifdef WSP
    #include "bandwidth.cuh"

    #define ZERO {DISTANCE_ZERO, BANDWIDTH_ZERO}
    #define UNITY {DISTANCE_UNITY, BANDWIDTH_UNITY}
#else
    #include "reliability.cuh"

    #define ZERO {DISTANCE_ZERO, RELIABILITY_ZERO}
    #define UNITY {DISTANCE_UNITY, RELIABILITY_UNITY}
#endif

__host__ __device__ lex_product plus(lex_product c1, lex_product c2);
__host__ __device__ lex_product times(lex_product c1, lex_product c2);
__host__ __device__ boolean less_eq(lex_product c1, lex_product c2);
__host__ __device__ boolean less(lex_product c1, lex_product c2); 
__host__ __device__ void lex_product_print(lex_product c);

#endif
