
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include "utilities.cuh"
#include "nset.cuh"
#include "path.cuh"
#include "pset.cuh"
#include "lex_product.cuh"
#include "wsp_or_mrsp.cuh"
#include "routing.cuh"

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
    for (unsigned int i = 0; i < n; i++)
        v_info[i] = i;
    
    a = (lex_product*) malloc(n*n * sizeof(lex_product));
    for (unsigned int i = 0; i < n*n; i++)
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
    for (unsigned int i = 0; i < n*n; i++)
        pset_free(pi[i]);
    free(pi);
    
    return 0;
}
