
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "node.h"
#include "reading_data.h"

void printA(double*** a, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
            printf("(%lf, %lf) ", a[i][j][0], a[i][j][1]);
        printf("\n");
    }
}

int main(void)
{
    int n_nodes;
    Node* v;
    double*** a;
    
    n_nodes = get_num_nodes("../../data/node.dat");
    if (n_nodes == ERR)
        exit(ERR);
    v = (Node*) malloc(sizeof(Node) * n_nodes);
    if(getV("../../data/node.dat", v) == ERR)
        exit(ERR);
    a = (double***) malloc(sizeof(double**) * n_nodes);
    for (int i = 0; i < n_nodes; i++)
        a[i] = (double**) malloc(sizeof(double*) * n_nodes);
    for (int i = 0; i < n_nodes; i++)
        for (int j = 0; j < n_nodes; j++)
        {
            a[i][j] = (double*) malloc(sizeof(double) * 2);
            a[i][j][0] = INFINITY;
            a[i][j][1] = 0;
        }
    if (getA("../../data/edge.dat", v, a) == -1)
        exit(ERR);

    printA(a, n_nodes);
    
    return 0;
}

// a host->device
