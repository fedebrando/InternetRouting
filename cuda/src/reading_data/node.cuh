
/*
 * Node
*/

#ifndef NODE
#define NODE

#include <stdio.h>
#include <math.h>
#include "lex_product.cuh"
#include "path.cuh"
#include "pset.cuh"

#define MAX_STRING_LENGTH 25

typedef struct
{
    char country[MAX_STRING_LENGTH];
    char label[MAX_STRING_LENGTH];
    char type[MAX_STRING_LENGTH];
    double latitude;
    double longitude;
} Node;

__host__ void Node_print(const Node* n);
__host__ double to_radians(double degree);
__host__ double haversine(const Node* n1, const Node* n2);

__host__ void print_results(const lex_product* d, pset** pi, const Node* v_info, size_t n);
__host__ void pset_v_info_print(const pset* s, const Node* v_info);
__host__ void path_v_info_print(const path* p, const Node* v_info);

#endif
