
/*
 * Lexicographic Product of two basic metrics
*/

#ifndef LEX_PRODUCT 
#define LEX_PRODUCT

#include <stdio.h>
#include "utilities.cuh"

typedef struct
{
    double fst;
    double snd;
} lex_product;

__host__ __device__ void lex_product_print(lex_product c);
__host__ __device__ boolean eq(lex_product c1, lex_product c2);

#endif
