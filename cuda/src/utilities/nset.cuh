
/*
 * Set of nodes 
*/

#ifndef NSET 
#define NSET

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utilities.cuh"

typedef struct
{
    size_t n_nodes;
    char* flags;
} nset;

__host__ __device__ size_t n_bytes(size_t n_nodes);
__host__ __device__ void nset_clear(nset* s);
__host__ nset* nset_create(size_t n_nodes);
__host__ __device__ boolean nset_in(const nset* s, node n);
__host__ __device__ void nset_print(const nset* s);
__host__ __device__ void nset_insert(nset* s, node n);
__host__ __device__ void nset_difference(nset* res, const nset* s1, const nset* s2);
__host__ void nset_free(nset* s);
__host__ nset* nset_host_to_device(const nset* s);
__host__ nset* nset_device_to_host(nset* s_dev);
__host__ void nset_free_device(nset* s_dev);

#endif
