
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "node.h"
#include "reading_data.h"
#include "lp.h"

#define DISTANCE_ZERO INFINITY
#define DISTANCE_UNITY 0.0

#ifdef SHORTEST_WIDEST
#define BANDWIDTH_ZERO 0.0
#define BANDWIDTH_UNITY INFINITY
#define ZERO ((lex_product) {DISTANCE_ZERO, BANDWIDTH_ZERO})
#define UNITY ((lex_product) {DISTANCE_UNITY, BANDWIDTH_UNITY})
#else
#define RELIABILITY_ZERO 0
#define RELIABILITY_UNITY 1
#define ZERO {DISTANCE_ZERO, RELIABILITY_ZERO}
#define UNITY {DISTANCE_UNITY, RELIABILITY_UNITY}
#endif

void printA(lex_product* a, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            lex_product_print(a[idx(i, j, dim)]);
            printf(" ");
        }
        printf("\n");
    }
}

int main(void)
{
    int n_nodes;
    Node* v;
    lex_product* a;
    
    n_nodes = get_num_nodes("../../data/node.dat");
    if (n_nodes == ERR)
        exit(ERR);
    v = (Node*) malloc(sizeof(Node) * n_nodes);
    if(getV("../../data/node.dat", v) == ERR)
        exit(ERR);
    a = (lex_product*) malloc(sizeof(lex_product) * n_nodes*n_nodes);
    for (int i = 0; i < n_nodes*n_nodes; i++)
        a[i] = ZERO;
    if (getA("../../data/edge.dat", v, a, n_nodes) == -1)
        exit(ERR);

    printA(a, n_nodes);
    
    return 0;
}

// a host->device
