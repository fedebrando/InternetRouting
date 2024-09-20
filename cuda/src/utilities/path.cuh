
/*
 * Path
*/

#ifndef PATH 
#define PATH

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "utilities.cuh"

typedef struct
{
    unsigned int n_nodes;
    unsigned int size;
    node* nodes;
} path;

__host__ __device__ path* path_create(unsigned int n_nodes);
__host__ void path_free(path* p);
__host__ __device__ boolean path_in(const path* p, node n);
__host__ __device__ boolean path_add(path* p, node i, node j);
__host__ __device__ boolean path_equal(const path* p1, const path* p2);
__host__ __device__ void path_print(const path* p);
__host__ path* path_host_to_device(const path* p);
__host__ path* path_device_to_host(path* p_dev);
__host__ void path_free_device(path* p_dev);

#endif
