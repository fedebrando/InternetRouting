
#include "lex_product.cuh"

__host__ __device__ boolean eq(lex_product c1, lex_product c2)
{
    return c1.fst == c2.fst && c1.snd == c2.snd;
}
