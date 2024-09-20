
#include "pset.cuh"

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
