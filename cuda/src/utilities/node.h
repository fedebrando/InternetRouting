
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

void Node_print(Node n);
double to_radians(double degree);
double haversine(Node n1, Node n2);

#endif
