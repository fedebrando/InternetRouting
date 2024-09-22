
#include "distance.cuh"

__host__ __device__ double distance_plus(double d1, double d2)
{
    return d1 <= d2 ? d1 : d2;
}

__host__ __device__ double distance_times(double d1, double d2)
{
    return d1 + d2;
}

__host__ __device__ boolean distance_less_eq(double d1, double d2)
{
    return d1 == distance_plus(d1, d2);
}

__host__ __device__ boolean distance_less(double d1, double d2)
{
    return d1 != d2 && distance_less_eq(d1, d2);
}

__host__ __device__ void distance_print(double d)
{
    printf("%lf km", d);
}
