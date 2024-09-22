
/*
 * Set of paths
*/

#ifndef PSET 
#define PSET

#include <stdio.h>
#include <stdlib.h>
#include "utilities.cuh"
#include "path.cuh"

// hyperparam to size the maximum expected optimal paths number
#define PROB_MAX_OPT_PATHS 5

typedef struct
{
    size_t n_nodes;
    size_t size;
    size_t m_size;
    path** paths;
} pset;

__host__ pset* pset_create(size_t n_nodes);
__host__ void pset_free(pset* s);
__host__ __device__ void pset_clear(pset* s);
__host__ __device__ boolean pset_in(const pset* s, const path* p);
__host__ __device__ void path_cpy(path* des, const path* src);
__host__ __device__ boolean pset_insert(pset* s, const path* p);
__host__ __device__ void pset_print(const pset* s);
__host__ __device__ boolean pset_pair_wise_concat(pset* res, const pset* s, node i, node j, path** paths_app);
__host__ pset* pset_host_to_device(const pset* s);
__host__ pset* pset_device_to_host(pset* s_dev);
__host__ void pset_free_device(pset* s_dev);

#endif
