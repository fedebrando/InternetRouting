
#ifndef READING_DATA
#define READING_DATA

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "node.h"

#define SHORTEST_WIDEST 

#define MAX_LINE_LENGTH 1024
#define MAX_STRING_LENGTH 25
#define ERR -1

int get_num_nodes(const char* filename);
int get_num_values(const char* line);
int getV(const char* filename, Node* v);
int getA(const char* filename, const Node* v, double*** a);

#endif
