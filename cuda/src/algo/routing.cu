
#include "routing.cuh"

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
