
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "utilities.cuh"
#include "nset.cuh"
#include "path.cuh"
#include "pset.cuh"
#include "lex_product.cuh"
#include "wsp_or_mrsp.cuh"
#include "routing.cuh"
#include "node.cuh"
#include "reading_data.cuh"

#define TIMING

#ifdef TIMING
#include <time.h>
#endif

int main(void)
{
#ifdef TIMING
    struct timespec start, end;
    double ms;
#endif
    unsigned int n;
    Node* v_info;
    lex_product *a;
    lex_product *d;
    pset **pi;

    // Reading file (Initialization)
    n = get_num_nodes("data/node.dat");
    if (n == 0)
    {
        fprintf(stderr, "Error: could not open file or empty file\n");
        exit(1);
    }

    v_info = (Node*) malloc(n * sizeof(Node));
    if (!getV("data/node.dat", v_info))
    {
        fprintf(stderr, "Error: could not open file\n");
        exit(1);
    }
    
    a = (lex_product*) malloc(n*n * sizeof(lex_product));
    if (!getA("data/edge.dat", v_info, a, n))
    {
        fprintf(stderr, "Error: could not open file\n");
        exit(1);
    }

#ifdef TIMING
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    // Parallel Routing Algorithm
    compute_routing(n, a, &d, &pi);
#ifdef TIMING
    clock_gettime(CLOCK_MONOTONIC, &end);
    ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
    printf("Elapsed time: %lf ms\n\n", ms);
#endif
    
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
