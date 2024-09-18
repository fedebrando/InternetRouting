
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

#define N 5

// Set of nodes *************************************************************************

typedef unsigned int node;

typedef struct
{
    unsigned int n_nodes;
    char* flags;
} nset;

__host__ __device__ unsigned int n_bytes(unsigned int n_nodes)
{
    return n_nodes / (8*sizeof(char)) + (n_nodes % (8*sizeof(char)) ? sizeof(char) : 0);
}

__host__ __device__ void nset_clear(nset* s)
{
    memset(s->flags, 0, n_bytes(s->n_nodes) * sizeof(char));
}

__host__ nset* nset_create(unsigned int n_nodes)
{
    nset* s = (nset*) malloc(sizeof(nset));
    
    s->n_nodes = n_nodes;
    s->flags = (char*) malloc(n_bytes(n_nodes) * sizeof(char));

    nset_clear(s);
    return s;
}

__host__ __device__ int nset_in(const nset* s, node n)
{
    return (int)s->flags[n/8] & (int)pow(2, n % 8);
}

__host__ __device__ void nset_print(const nset* s)
{
    int empty = 1;

    printf("{");
    for (node n = 0; n < s->n_nodes; n++)
        if (nset_in(s, n))
        {
            printf("%d, ", n);
            if (empty)
                empty = 0;
        }
    if (!empty)
        printf("\b\b");
    printf("}\n");
}

__host__ __device__ void nset_insert(nset* s, node n)
{
    s->flags[n/8] = (unsigned int)s->flags[n/8] | (unsigned int)pow(2, n % 8);
}

__host__ __device__ void nset_difference(nset* res, const nset* s1, const nset* s2)
{
    nset_clear(res);
    for (node n = 0; n < N; n++)
        if (nset_in(s1, n) && !nset_in(s2, n))
            nset_insert(res, n);
}

__host__ void nset_free(nset* s)
{
    free(s->flags);
    free(s);
}

__host__ nset* nset_host_to_device(const nset* s)
{
    char* flags_dev;
    nset* s_dev;
    unsigned int flags_size = n_bytes(s->n_nodes);
    nset s_app;

    cudaMalloc(&flags_dev, flags_size * sizeof(char));
    cudaMemcpy(flags_dev, s->flags, flags_size * sizeof(char), cudaMemcpyHostToDevice);

    s_app.n_nodes = s->n_nodes;
    s_app.flags = flags_dev;
    cudaMalloc(&s_dev, sizeof(nset));
    cudaMemcpy(s_dev, &s_app, sizeof(nset), cudaMemcpyHostToDevice);
    
    return s_dev;
}

__host__ nset* nset_device_to_host(nset* s_dev)
{
    nset* s = (nset*) malloc(sizeof(nset));
    char* flags;
    unsigned int flags_size;

    cudaMemcpy(s, s_dev, sizeof(nset), cudaMemcpyDeviceToHost);
    cudaFree(s_dev);
    flags_size = n_bytes(s->n_nodes);
    flags = (char*) malloc(flags_size * sizeof(char));
    cudaMemcpy(flags, s->flags, flags_size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(s->flags);
    s->flags = flags;
    return s;
}

__host__ void nset_free_device(nset* s_dev)
{
    nset* s = nset_device_to_host(s_dev);

    nset_free(s);
}

// END Set of nodes *************************************************************************

// Path *************************************************************************************

typedef struct
{
    unsigned int n_nodes;
    int size;
    node* nodes;
} path;

__host__ __device__ path* path_create(unsigned int n_nodes)
{
    path* p = (path*) malloc(sizeof(path));

    p->size = 0;
    p->n_nodes = n_nodes;
    p->nodes = (node*) malloc(n_nodes * sizeof(node));
    return p;
}

__host__ void path_free(path* p)
{
    free(p->nodes);
    free(p);
}

__host__ __device__ int path_in(const path* p, node n)
{
    for (int i = 0; i < p->size; i++)
        if (p->nodes[i] == n)
            return 1;
    return 0;
}

__host__ __device__ int path_add(path* p, node i, node j)
{
    if (p->size)
    {
        if (p->nodes[p->size - 1] != i)
            return 0;
        if (path_in(p, j))
            return 0;
        p->nodes[p->size] = j;
        p->size++;
    }
    else
    {
        p->nodes[0] = i;
        p->nodes[1] = j;
        p->size += 2;
    }
    return 1;
}

__host__ __device__ int path_equal(const path* p1, const path* p2)
{
    if (p1->size != p2->size)
        return 0;
    for (int i = 0; i < p1->size; i++)
        if (p1->nodes[i] != p2->nodes[i])
            return 0;
    return 1;
}

__host__ __device__ void path_print(const path* p)
{
    if (p->size)
    {
        printf("(");
        for (int i = 0; i < p->size; i++)
            printf("%d, ", p->nodes[i]);
        printf("\b\b)");
    }
    else
        printf("ε");
}

__host__ path* path_host_to_device(const path* p)
{
    path* p_dev;
    node* nodes_dev;
    path p_app;

    cudaMalloc(&nodes_dev, p->n_nodes * sizeof(node));
    cudaMemcpy(nodes_dev, p->nodes, p->n_nodes * sizeof(node), cudaMemcpyHostToDevice);

    p_app.n_nodes = p->n_nodes;
    p_app.size = p->size;
    p_app.nodes = nodes_dev;
    cudaMalloc(&p_dev, sizeof(path));
    cudaMemcpy(p_dev, &p_app, sizeof(path), cudaMemcpyHostToDevice);
    return p_dev;
}

__host__ path* path_device_to_host(path* p_dev)
{
    path* p = (path*) malloc(sizeof(path));
    node* nodes;

    cudaMemcpy(p, p_dev, sizeof(path), cudaMemcpyDeviceToHost);
    cudaFree(p_dev);
    nodes = (node*) malloc(p->n_nodes * sizeof(node));
    cudaMemcpy(nodes, p->nodes, p->n_nodes * sizeof(node), cudaMemcpyDeviceToHost);
    cudaFree(p->nodes);
    p->nodes = nodes;
    return p;
}

__host__ void path_free_device(path* p_dev)
{
    path* p = path_device_to_host(p_dev);

    path_free(p);
}

// END Path *********************************************************************************

// Set of Paths *****************************************************************************

#define PROB_MAX_OPT_PATHS 5

typedef struct
{
    unsigned int n_nodes;
    int size;
    int m_size;
    path** paths;
} pset;

__host__ __device__ pset* pset_create(unsigned int n_nodes)
{
    pset* s = (pset*) malloc(sizeof(pset));

    s->n_nodes = n_nodes;
    s->size = 0;
    s->m_size = PROB_MAX_OPT_PATHS;
    s->paths = (path**) malloc(s->m_size * sizeof(path*));
    for (int i = 0; i < s->m_size; i++)
        s->paths[i] = path_create(n_nodes);
    return s;
}

__host__ void pset_free(pset* s)
{
    for (int i = 0; i < s->m_size; i++)
        path_free(s->paths[i]);
    free(s->paths);
    free(s);
}

__host__ __device__ void pset_clear(pset* s)
{
    s->size = 0;
}

__host__ __device__ int pset_in(const pset* s, const path* p)
{
    for (int i = 0; i < s->size; i++)
        if (path_equal(s->paths[i], p))
            return 1;
    return 0;
}

__device__ void* realloc_dev(void* old_ptr, size_t old_size, size_t new_size)
{
    void* new_ptr;

    if (new_size < old_size)
        return NULL;
    else if (new_size == old_size)
        return old_ptr;
    new_ptr = malloc(new_size);
    memcpy(new_ptr, old_ptr, old_size);
    free(old_ptr);
    return new_ptr;
}

__device__ void path_cpy(path* des, const path* src)
{
    des->n_nodes = src->n_nodes;
    des->size = src->size;
    memcpy(des->nodes, src->nodes, src->n_nodes * sizeof(node));
}

__device__ void pset_insert(pset* s, const path* p)
{
    if (s->size < s->m_size)
    {
        if (pset_in(s, p))
            return;
        path_cpy(s->paths[s->size], p);
        s->size++;
    }
    else
    {
        s->paths = (path**) realloc_dev(s->paths, s->m_size * sizeof(path*), (s->m_size + 1) * sizeof(path*));
        s->paths[s->m_size] = path_create(s->n_nodes);
        s->m_size++;
        pset_insert(s, p);
    }
}

__host__ __device__ void pset_print(const pset* s)
{
    printf("{");
    for (int i = 0; i < s->size; i++)
    {
        path_print(s->paths[i]);
        if (i != s->size - 1)
            printf(", ");
    }
    printf("}");
}

__device__ void pset_pair_wise_concat(pset* res, const pset* s, node i, node j, path** paths_app)
{
    for (int p = 0; p < s->size; p++)
    {
        path_cpy(paths_app[p], s->paths[p]);
        if (path_add(paths_app[p], i, j))
            pset_insert(res, paths_app[p]);
    }
}

__host__ pset* pset_host_to_device(const pset* s)
{
    pset* s_dev;
    pset s_app;
    path** paths_dev;
    path** paths_ptr_dev = (path**) malloc(s->m_size * sizeof(path*));

    for (int i = 0; i < s->m_size; i++)
        paths_ptr_dev[i] = path_host_to_device(s->paths[i]);

    cudaMalloc(&paths_dev, s->m_size * sizeof(path*));
    cudaMemcpy(paths_dev, paths_ptr_dev, s->m_size * sizeof(path*), cudaMemcpyHostToDevice);

    s_app.n_nodes = s->n_nodes;
    s_app.size = s->size;
    s_app.m_size = s->m_size;
    s_app.paths = paths_dev;
    cudaMalloc(&s_dev, sizeof(pset));
    cudaMemcpy(s_dev, &s_app, sizeof(pset), cudaMemcpyHostToDevice);

    free(paths_ptr_dev);

    return s_dev;
}

__host__ pset* pset_device_to_host(pset* s_dev)
{
    pset* s = (pset*) malloc(sizeof(pset));
    path** paths_ptr_dev;
    path** paths;

    cudaMemcpy(s, s_dev, sizeof(pset), cudaMemcpyDeviceToHost);
    cudaFree(s_dev);

    paths_ptr_dev = (path**) malloc(s->m_size * sizeof(path*));
    cudaMemcpy(paths_ptr_dev, s->paths, s->m_size * sizeof(path*), cudaMemcpyDeviceToHost);
    cudaFree(s->paths);

    paths = (path**) malloc(s->m_size * sizeof(path*));
    for (int i = 0; i < s->m_size; i++)
        paths[i] = path_device_to_host(paths_ptr_dev[i]);
    free(paths_ptr_dev);

    s->paths = paths;

    return s;
}

__host__ void pset_free_device(pset* s_dev)
{
    pset* s = pset_device_to_host(s_dev);

    pset_free(s);
}

// END Set of Paths *************************************************************************

// Distance *******************

#define DISTANCE_ZERO INFINITY
#define DISTANCE_UNITY 0

__host__ __device__ double distance_plus(double d1, double d2)
{
    return d1 <= d2 ? d1 : d2;
}

__host__ __device__ double distance_times(double d1, double d2)
{
    return d1 + d2;
}

__host__ __device__ int distance_less_eq(double d1, double d2)
{
    return d1 == distance_plus(d1, d2);
}

__host__ __device__ int distance_less(double d1, double d2)
{
    return d1 != d2 && distance_less_eq(d1, d2);
}

// Reliability *******************

#define RELIABILITY_ZERO 0
#define RELIABILITY_UNITY 1

__host__ __device__ double reliability_plus(double r1, double r2)
{
    return r1 >= r2 ? r1 : r2;
}

__host__ __device__ double reliability_times(double r1, double r2)
{
    return r1 * r2;
}

__host__ __device__ int reliability_less_eq(double r1, double r2)
{
    return r1 == reliability_plus(r1, r2);
}

__host__ __device__ int reliability_less(double r1, double r2)
{
    return r1 != r2 && reliability_less_eq(r1, r2);
}

// Reliability *******************

#define BANDWIDTH_ZERO 0
#define BANDWIDTH_UNITY INFINITY

__host__ __device__ double bandwidth_plus(double bw1, double bw2)
{
    return bw1 >= bw2 ? bw1 : bw2;
}

__host__ __device__ double bandwidth_times(double bw1, double bw2)
{
    return bw1 <= bw2 ? bw1 : bw2;
}

__host__ __device__ int bandwidth_less_eq(double bw1, double bw2)
{
    return bw1 == bandwidth_plus(bw1, bw2);
}

__host__ __device__ int bandwidth_less(double bw1, double bw2)
{
    return bw1 != bw2 && bandwidth_less_eq(bw1, bw2);
}

// lex_product ***********************************************

typedef struct
{
    double fst;
    double snd;
} lex_product;

__host__ __device__ void lex_product_print(lex_product c)
{
    printf("(%lf, %lf)", c.fst, c.snd);
}

__host__ __device__ int eq(lex_product c1, lex_product c2)
{
    return c1.fst == c2.fst && c1.snd == c2.snd;
}

/*
// LexProduct<Distance, Reliability>

#define ZERO {DISTANCE_ZERO, RELIABILITY_ZERO}
#define UNITY {DISTANCE_UNITY, RELIABILITY_UNITY}

__host__ __device__ lex_product plus(lex_product c1, lex_product c2)
{
    if (distance_less(c1.fst, c2.fst))
        return c1;
    if (distance_less(c2.fst, c1.fst))
        return c2;
    return (lex_product) {c1.fst, reliability_plus(c1.snd, c2.snd)};
}

__host__ __device__ lex_product times(lex_product c1, lex_product c2)
{
    return (lex_product) {distance_times(c1.fst, c2.fst), reliability_times(c1.snd, c2.snd)};
}

*/

// LexProduct<Distance, Bandwidth>

#define ZERO {DISTANCE_ZERO, BANDWIDTH_ZERO}
#define UNITY {DISTANCE_UNITY, BANDWIDTH_UNITY}

__host__ __device__ lex_product plus(lex_product c1, lex_product c2)
{
    if (distance_less(c1.fst, c2.fst))
        return c1;
    if (distance_less(c2.fst, c1.fst))
        return c2;
    return (lex_product) {c1.fst, bandwidth_plus(c1.snd, c2.snd)};
}

__host__ __device__ lex_product times(lex_product c1, lex_product c2)
{
    return (lex_product) {distance_times(c1.fst, c2.fst), bandwidth_times(c1.snd, c2.snd)};
}

// *************************************************************

// Order func ********************************************

__host__ __device__ int less_eq(lex_product c1, lex_product c2)
{
    return eq(c1, plus(c1, c2));
}

__host__ __device__ int less(lex_product c1, lex_product c2)
{
    return !eq(c1, c2) && less_eq(c1, c2);
}

// ***********************************

__host__ __device__ unsigned int idx(unsigned int i, unsigned int j, unsigned int n)
{
    return i*n + j;
}

__device__ node find_qk_min(const lex_product* d, const nset* v, nset** s, nset** diff, int i, int n)
{
    node qk;
    lex_product qk_min = ZERO;

    nset_difference(diff[i], v, s[i]);
    for (node j = 0; j < n; j++)
    {
        if (!nset_in(diff[i], j))
            continue;

        if (less_eq(d[idx(i, j, n)], qk_min))
        {
            qk = j;
            qk_min = d[idx(i, j, n)];
        }
    }
    return qk;
}

__global__ void dijkstra(const lex_product* a, lex_product* d, pset** pi, const path* eps, int n, const nset* v, nset** s, nset** diff, path*** paths_app)
{
    int i = threadIdx.x;
    node qk;

    for (node q = 0; q < n; q++)
        d[idx(i, q, n)] = ZERO;
    d[idx(i, i, n)] = UNITY;
    pset_insert(pi[idx(i, i, n)], eps);

    for (node k = 0; k < n; k++)
    {
        qk = find_qk_min(d, v, s, diff, i, n);
        nset_insert(s[i], qk);
        nset_difference(diff[i], v, s[i]);
        for (node j = 0; j < n; j++)
        {
            if (!nset_in(diff[i], j))
                continue;

            if (eq(times(d[idx(i, qk, n)], a[idx(qk, j, n)]), d[idx(i, j, n)]))
            {
                pset_pair_wise_concat(pi[idx(i, j, n)], pi[idx(i, qk, n)], qk, j, paths_app[i]);
            }
            else if (less(times(d[idx(i, qk, n)], a[idx(qk, j, n)]), d[idx(i, j, n)]))
            {
                d[idx(i, j, n)] = times(d[idx(i, qk, n)], a[idx(qk, j, n)]);

                pset_clear(pi[idx(i, j, n)]);
                pset_pair_wise_concat(pi[idx(i, j, n)], pi[idx(i, qk, n)], qk, j, paths_app[i]);
            }
        }
    }
}

__host__ nset* v_host_to_device(unsigned int n)
{
    nset *v, *v_dev;

    v = nset_create(n);
    for (node q = 0; q < n; q++)
        nset_insert(v, q);
    v_dev = nset_host_to_device(v);
    nset_free(v);

    return v_dev;
}

__host__ nset** s_host_to_device(unsigned int n)
{
    nset *s, **s_dev_ptr, **s_dev;

    s = nset_create(n);
    s_dev_ptr = (nset**) malloc(n * sizeof(nset*));
    for (int i = 0; i < n; i++)
        s_dev_ptr[i] = nset_host_to_device(s);
    nset_free(s);
    cudaMalloc(&s_dev, n * sizeof(nset*));
    cudaMemcpy(s_dev, s_dev_ptr, n * sizeof(nset*), cudaMemcpyHostToDevice);
    free(s_dev_ptr);
    return s_dev;
}

__host__ void s_free_device(unsigned int n, nset** s_dev)
{
    nset** s_dev_ptr = (nset**) malloc(n * sizeof(nset*));

    cudaMemcpy(s_dev_ptr, s_dev, n * sizeof(nset*), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        nset_free_device(s_dev_ptr[i]);
    cudaFree(s_dev);
    free(s_dev_ptr);
}

__host__ nset** diff_host_to_device(unsigned int n)
{
    nset *diff, **diff_dev_ptr, **diff_dev;

    diff = nset_create(n);
    diff_dev_ptr = (nset**) malloc(n * sizeof(nset*));
    for (int i = 0; i < n; i++)
        diff_dev_ptr[i] = nset_host_to_device(diff);
    nset_free(diff);
    cudaMalloc(&diff_dev, n * sizeof(nset*));
    cudaMemcpy(diff_dev, diff_dev_ptr, n * sizeof(nset*), cudaMemcpyHostToDevice);
    free(diff_dev_ptr);
    return diff_dev;
}

__host__ void diff_free_device(unsigned int n, nset** diff_dev)
{
    nset** diff_dev_ptr = (nset**) malloc(n * sizeof(nset*));

    cudaMemcpy(diff_dev_ptr, diff_dev, n * sizeof(nset*), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        nset_free_device(diff_dev_ptr[i]);
    cudaFree(diff_dev);
    free(diff_dev_ptr);
}

__host__ pset** pi_host_to_device(unsigned int n)
{
    pset **pi, **pi_dev_ptr, **pi_dev;

    pi = (pset**) malloc(n*n * sizeof(pset*));
    for (int i = 0; i < n*n; i++)
        pi[i] = pset_create(n);
    pi_dev_ptr = (pset**) malloc(n*n * sizeof(pset*));
    for (int i = 0; i < n*n; i++)
        pi_dev_ptr[i] = pset_host_to_device(pi[i]);
    cudaMalloc(&pi_dev, n*n * sizeof(pset*));
    cudaMemcpy(pi_dev, pi_dev_ptr, n*n * sizeof(pset*), cudaMemcpyHostToDevice);

    free(pi_dev_ptr);
    for (int i = 0; i < n*n; i++)
        pset_free(pi[i]);
    free(pi);

    return pi_dev;
}

__host__ pset** pi_device_to_host(unsigned int n, pset** pi_dev)
{
    pset **pi_dev_ptr, **pi_res;

    pi_res = (pset**) malloc(n*n * sizeof(pset*));
    pi_dev_ptr = (pset**) malloc(n*n * sizeof(pset*));
    cudaMemcpy(pi_dev_ptr, pi_dev, n*n * sizeof(path*), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n*n; i++)
        pi_res[i] = pset_device_to_host(pi_dev_ptr[i]);

    free(pi_dev_ptr);

    return pi_res;
}

__host__ path** paths_app_host_to_device(unsigned int n)
{
    path* p = path_create(n);
    path** paths_app_dev_ptr = (path**) malloc(n * sizeof(path*));
    path** paths_app_dev;

    for (int i = 0; i < n; i++)
        paths_app_dev_ptr[i] = path_host_to_device(p);
    cudaMalloc(&paths_app_dev, n * sizeof(path*));
    cudaMemcpy(paths_app_dev, paths_app_dev_ptr, n * sizeof(path*), cudaMemcpyHostToDevice);

    free(paths_app_dev_ptr);
    path_free(p);

    return paths_app_dev;
}

__host__ void paths_app_free_device(unsigned int n, path** paths_app_dev)
{
    path** paths_app_dev_ptr = (path**) malloc(n * sizeof(path*));

    cudaMemcpy(paths_app_dev_ptr, paths_app_dev, n * sizeof(path*), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        path_free_device(paths_app_dev_ptr[i]);
    cudaFree(paths_app_dev);

    free(paths_app_dev_ptr);
}

__host__ path*** multi_paths_app_host_to_device(unsigned int n)
{
    path*** mpa_dev;
    path*** mpa_dev_ptr = (path***) malloc(n * sizeof(path**));

    for (int i = 0; i < n; i++)
        mpa_dev_ptr[i] = paths_app_host_to_device(n);
    cudaMalloc(&mpa_dev, n * sizeof(path**));
    cudaMemcpy(mpa_dev, mpa_dev_ptr, n * sizeof(path**), cudaMemcpyHostToDevice);

    free(mpa_dev_ptr);

    return mpa_dev;
}

__host__ void multi_paths_app_free_device(unsigned int n, path*** mpa_dev)
{
    path*** mpa_dev_ptr = (path***) (malloc(n * sizeof(path**)));

    cudaMemcpy(mpa_dev_ptr, mpa_dev, n * sizeof(path**), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        paths_app_free_device(n, mpa_dev_ptr[i]);
    cudaFree(mpa_dev);

    free(mpa_dev_ptr);
}

__host__ __device__ void d_print(unsigned int n, const lex_product* d)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            lex_product_print(d[idx(i, j, n)]);
            printf(" ");
        }
        printf("\n");
    }
}

__host__ __device__ void pi_print(unsigned int n, pset** pi)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            pset_print(pi[idx(i, j, n)]);
            printf(" ");
        }
        printf("\n");
    }
}

__host__ void compute_routing(unsigned int n, const lex_product* a, lex_product** d, pset*** pi)
{
    nset* v_dev;
    lex_product *a_dev, *d_dev;
    nset **s_dev, **diff_dev;
    pset** pi_dev;
    path *eps, *eps_dev, ***paths_app_dev;

    // Host -> Device
    cudaMalloc(&a_dev, n*n * sizeof(lex_product)); // a_dev
    cudaMemcpy(a_dev, a, n*n * sizeof(lex_product), cudaMemcpyHostToDevice);
    v_dev = v_host_to_device(n); // v_dev
    cudaMalloc(&d_dev, n*n * sizeof(lex_product)); // d_dev
    s_dev = s_host_to_device(n); // s_dev
    diff_dev = diff_host_to_device(n); // diff_dev
    pi_dev = pi_host_to_device(n); // pi_dev
    eps = path_create(n);
    eps_dev = path_host_to_device(eps); // eps_dev
    paths_app_dev = multi_paths_app_host_to_device(n);

    // Parallel Computing
    dijkstra<<<1, n>>>(a_dev, d_dev, pi_dev, eps_dev, n, v_dev, s_dev, diff_dev, paths_app_dev);
    cudaDeviceSynchronize();

    // Device -> Host (Results)
    *d = (lex_product*) malloc(n*n * sizeof(lex_product));
    cudaMemcpy(*d, d_dev, n*n * sizeof(lex_product), cudaMemcpyDeviceToHost); // d
    *pi = pi_device_to_host(n, pi_dev); // pi

    // Free
    path_free(eps);
    nset_free_device(v_dev);
    cudaFree(a_dev);
    cudaFree(d_dev);
    s_free_device(n, s_dev);
    diff_free_device(n, diff_dev);
    path_free_device(eps_dev);
    cudaFree(pi_dev);
    multi_paths_app_free_device(n, paths_app_dev);
}

int main(void)
{
    unsigned int n;
    node* v_info;
    lex_product *a;
    lex_product *d;
    pset **pi;

    // Reading file (Initialization)
    n = 5;
    v_info = (node*) malloc(n * sizeof(node));
    for (int i = 0; i < n; i++)
        v_info[i] = i;
    
    a = (lex_product*) malloc(n*n * sizeof(lex_product));
    for (int i = 0; i < n*n; i++)
        a[i] = ZERO;
    /*
    a[idx(0, 1, n)] = {1, 0.5};
    a[idx(1, 0, n)] = {1, 0.5};
    a[idx(1, 4, n)] = {1, 0.1};
    a[idx(4, 1, n)] = {1, 0.1};
    a[idx(4, 2, n)] = {1, 1};
    a[idx(2, 4, n)] = {1, 1};
    a[idx(1, 2, n)] = {2, 0.81};
    a[idx(2, 1, n)] = {2, 0.81};
    a[idx(1, 3, n)] = {1, 0.9};
    a[idx(3, 1, n)] = {1, 0.9};
    a[idx(3, 2, n)] = {1, 0.9};
    a[idx(2, 3, n)] = {1, 0.9};
    */
    a[idx(0, 1, n)] = {1, 10};
    a[idx(1, 0, n)] = {1, 10};
    a[idx(1, 4, n)] = {1, 100};
    a[idx(4, 1, n)] = {1, 100};
    a[idx(4, 2, n)] = {1, 100};
    a[idx(2, 4, n)] = {1, 100};
    a[idx(1, 2, n)] = {2, 90};
    a[idx(2, 1, n)] = {2, 90};
    a[idx(1, 3, n)] = {1, 5};
    a[idx(3, 1, n)] = {1, 5};
    a[idx(3, 2, n)] = {1, 100};
    a[idx(2, 3, n)] = {1, 100};

    // Parallel Routing Algorithm
    compute_routing(n, a, &d, &pi);

    // Results printing
    printf("--- D MATRIX ---\n");
    d_print(n, d);

    printf("\n--- PI MATRIX ---\n");
    pi_print(n, pi);

    // Free
    free(v_info);
    free(a);
    free(d);
    for (int i = 0; i < n*n; i++)
        pset_free(pi[i]);
    free(pi);
    
    return 0;
}
