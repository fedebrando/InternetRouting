
#ifndef UTILITIES 
#define UTILITIES

#include <stddef.h>

typedef size_t node;
typedef short int boolean;

__host__ __device__ size_t idx(node i, node j, size_t n);

#endif
