
#include "path.cuh"

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

__host__ __device__ boolean path_in(const path* p, node n)
{
    for (node i = 0; i < p->size; i++)
        if (p->nodes[i] == n)
            return 1;
    return 0;
}

__host__ __device__ boolean path_add(path* p, node i, node j)
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

__host__ __device__ boolean path_equal(const path* p1, const path* p2)
{
    if (p1->size != p2->size)
        return 0;
    for (node i = 0; i < p1->size; i++)
        if (p1->nodes[i] != p2->nodes[i])
            return 0;
    return 1;
}

__host__ __device__ void path_print(const path* p)
{
    if (p->size)
    {
        printf("(");
        for (node i = 0; i < p->size; i++)
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
