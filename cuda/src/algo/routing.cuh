
/*
 * Routing Algorithm
*/

#ifndef ROUTING 
#define ROUTING

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "utilities.cuh"
#include "nset.cuh"
#include "path.cuh"
#include "pset.cuh"
#include "lex_product.cuh"
#include "wsp_or_mrsp.cuh"

#define TIMING

#ifdef TIMING
#include <time.h>
#endif

__device__ node find_qk_min(const lex_product* d, const nset* v, nset** s, nset** diff, node i, unsigned int n);
__global__ void dijkstra(boolean* err, const lex_product* a, lex_product* d, pset** pi, const path* eps, unsigned int n, const nset* v, nset** s, nset** diff, path*** paths_app);
__host__ nset* v_host_to_device(unsigned int n);
__host__ nset** s_host_to_device(unsigned int n);
__host__ void s_free_device(unsigned int n, nset** s_dev);
__host__ nset** diff_host_to_device(unsigned int n);
__host__ void diff_free_device(unsigned int n, nset** diff_dev);
__host__ pset** pi_host_to_device(unsigned int n);
__host__ pset** pi_device_to_host(unsigned int n, pset** pi_dev);
__host__ path** paths_app_host_to_device(unsigned int n);
__host__ void paths_app_free_device(unsigned int n, path** paths_app_dev);
__host__ path*** multi_paths_app_host_to_device(unsigned int n);
__host__ void multi_paths_app_free_device(unsigned int n, path*** mpa_dev);
__host__ boolean* err_host_to_device(unsigned int n);
__host__ boolean check_error(unsigned int n, const boolean* err);
__host__ __device__ void d_print(unsigned int n, const lex_product* d);
__host__ __device__ void pi_print(unsigned int n, pset** pi);
__host__ void compute_routing(unsigned int n, const lex_product* a, lex_product** d, pset*** pi);

#endif
