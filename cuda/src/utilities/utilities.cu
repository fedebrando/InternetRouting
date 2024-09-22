
#include "utilities.cuh"

__host__ __device__ size_t idx(node i, node j, size_t n)
{
    return i*n + j;
}
