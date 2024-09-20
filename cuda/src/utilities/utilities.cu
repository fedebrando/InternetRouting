
#include "utilities.cuh"

__host__ __device__ unsigned int idx(unsigned int i, unsigned int j, unsigned int n)
{
    return i*n + j;
}
