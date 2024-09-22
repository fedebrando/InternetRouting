
/*
 * Lexicographic Product of two basic metrics
*/

#ifndef LEX_PRODUCT 
#define LEX_PRODUCT

#include "utilities.cuh"

typedef struct
{
    double fst;
    double snd;
} lex_product;

__host__ __device__ boolean eq(lex_product c1, lex_product c2);

#endif
