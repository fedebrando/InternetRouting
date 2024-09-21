
#include "nset.cuh"

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

__host__ __device__ boolean nset_in(const nset* s, node n)
{
    return (int)s->flags[n/8] & (int)pow(2, n % 8);
}

__host__ __device__ void nset_print(const nset* s)
{
    printf("{");
    for (node n = 0; n < s->n_nodes; n++)
        if (nset_in(s, n))
        {
            if (n == s->n_nodes - 1)
                printf("%d", n);
            else
                printf("%d, ", n);
        }
    printf("}");
}

__host__ __device__ void nset_insert(nset* s, node n)
{
    s->flags[n/8] = (unsigned int)s->flags[n/8] | (unsigned int)pow(2, n % 8);
}

__host__ __device__ void nset_difference(nset* res, const nset* s1, const nset* s2)
{
    nset_clear(res);
    for (node n = 0; n < s1->n_nodes; n++)
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
