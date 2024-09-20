
#ifndef READING_DATA
#define READING_DATA

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "utilities.cuh"
#include "node.cuh"
#include "lex_product.cuh"

#define MAX_LINE_LENGTH 1024
#define MAX_STRING_LENGTH 25
#define ERR -1

__host__ int get_num_nodes(const char* filename);
__host__ int get_num_values(const char* line);
__host__ int getV(const char* filename, Node* v);
__host__ int getA(const char* filename, const Node* v, lex_product* a, int n);

#endif
