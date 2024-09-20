
#include "utilities.cuh"

__host__ __device__ unsigned int idx(node i, node j, unsigned int n)
{
    return i*n + j;
}
