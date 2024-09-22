
/*
 * Node
*/

#ifndef NODE
#define NODE

#include <stdio.h>
#include <math.h>

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

#endif
