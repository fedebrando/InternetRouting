
#ifndef LP
#define LP

#include <stdio.h>

typedef struct
{
    double fst;
    double snd;
} lex_product;
 
void lex_product_print(lex_product c);
unsigned int idx(unsigned int i, unsigned int j, unsigned int n);

#endif
