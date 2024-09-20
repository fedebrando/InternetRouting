
#include "reliability.cuh"

__host__ __device__ double reliability_plus(double r1, double r2)
{
    return r1 >= r2 ? r1 : r2;
}

__host__ __device__ double reliability_times(double r1, double r2)
{
    return r1 * r2;
}

__host__ __device__ int reliability_less_eq(double r1, double r2)
{
    return r1 == reliability_plus(r1, r2);
}

__host__ __device__ int reliability_less(double r1, double r2)
{
    return r1 != r2 && reliability_less_eq(r1, r2);
}
