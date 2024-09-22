
#include "wsp_or_mrsp.cuh"

__host__ __device__ lex_product plus(lex_product c1, lex_product c2)
{
    if (distance_less(c1.fst, c2.fst))
        return c1;
    if (distance_less(c2.fst, c1.fst))
        return c2;
#ifdef WSP
    return (lex_product) {c1.fst, bandwidth_plus(c1.snd, c2.snd)};
#else
    return (lex_product) {c1.fst, reliability_plus(c1.snd, c2.snd)};
#endif
}

__host__ __device__ lex_product times(lex_product c1, lex_product c2)
{
#ifdef WSP
    return (lex_product) {distance_times(c1.fst, c2.fst), bandwidth_times(c1.snd, c2.snd)};
#else
    return (lex_product) {distance_times(c1.fst, c2.fst), reliability_times(c1.snd, c2.snd)};
#endif
}

__host__ __device__ boolean less_eq(lex_product c1, lex_product c2)
{
    return eq(c1, plus(c1, c2));
}

__host__ __device__ boolean less(lex_product c1, lex_product c2)
{
    return !eq(c1, c2) && less_eq(c1, c2);
}

__host__ __device__ void lex_product_print(lex_product c)
{
    printf("(");
    distance_print(c.fst);
    printf(", ");
#ifdef WSP
    bandwidth_print(c.snd);
#else
    reliability_print(c.snd);
#endif
    printf(")");
}
