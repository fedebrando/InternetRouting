
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

int main(void)
{
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
    for (int i = 0; i < n*n; i++)
        a[i] = ZERO;
    if (!getA("data/edge.dat", v_info, a, n))
    {
        fprintf(stderr, "Error: could not open file\n");
        exit(1);
    }
    
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
