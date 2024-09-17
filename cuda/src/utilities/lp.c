
#include "lp.h"

void lex_product_print(lex_product c)
{
    printf("(%lf, %lf)", c.fst, c.snd);
}

unsigned int idx(unsigned int i, unsigned int j, unsigned int n)
{
    return i*n + j;
}
