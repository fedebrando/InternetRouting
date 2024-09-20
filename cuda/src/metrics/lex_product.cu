
#include "lex_product.cuh"

__host__ __device__ void lex_product_print(lex_product c)
{
    printf("(%lf, %lf)", c.fst, c.snd);
}

__host__ __device__ boolean eq(lex_product c1, lex_product c2)
{
    return c1.fst == c2.fst && c1.snd == c2.snd;
}
