
#include "bandwidth.cuh"

__host__ __device__ double bandwidth_plus(double bw1, double bw2)
{
    return bw1 >= bw2 ? bw1 : bw2;
}

__host__ __device__ double bandwidth_times(double bw1, double bw2)
{
    return bw1 <= bw2 ? bw1 : bw2;
}

__host__ __device__ int bandwidth_less_eq(double bw1, double bw2)
{
    return bw1 == bandwidth_plus(bw1, bw2);
}

__host__ __device__ int bandwidth_less(double bw1, double bw2)
{
    return bw1 != bw2 && bandwidth_less_eq(bw1, bw2);
}
