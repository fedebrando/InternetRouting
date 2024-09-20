
/*
 * Set of path
*/

#ifndef PSET 
#define PSET

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "utilities.cuh"
#include "path.cuh"

// hyperparam to size the maximum expected optimal paths number
#define PROB_MAX_OPT_PATHS 1

typedef struct
{
    unsigned int n_nodes;
    int size;
    int m_size;
    path** paths;
} pset;

__host__ __device__ pset* pset_create(unsigned int n_nodes);
__host__ void pset_free(pset* s);
__host__ __device__ void pset_clear(pset* s);
__host__ __device__ int pset_in(const pset* s, const path* p);
__device__ void* realloc_dev(void* old_ptr, size_t old_size, size_t new_size);
__device__ void path_cpy(path* des, const path* src);
__device__ void pset_insert(pset* s, const path* p);
__host__ __device__ void pset_print(const pset* s);
__device__ void pset_pair_wise_concat(pset* res, const pset* s, node i, node j, path** paths_app);
__host__ pset* pset_host_to_device(const pset* s);
__host__ pset* pset_device_to_host(pset* s_dev);
__host__ void pset_free_device(pset* s_dev);

#endif
