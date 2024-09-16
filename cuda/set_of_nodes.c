
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define N 9
#define DIM_FLAGS (N / (8*sizeof(char)) + (N % (8*sizeof(char)) ? sizeof(char) : 0))

typedef unsigned int node;
typedef int bool;
#define true (1 == 1)
#define false (0 == 1)

typedef struct
{
    char flags[DIM_FLAGS];
} set;

bool check_node(node n)
{
    return 0 <= n && n <= N-1;
}

set* set_create()
{
    set* s = (set*) malloc(sizeof(set));

    memset(s->flags, 0, DIM_FLAGS*sizeof(char));
    return s;
}

bool set_in(const set* s, node n)
{
    return (int)s->flags[n/8] & (int)pow(2, n % 8);
}

void set_print(const set* s)
{
    int empty = 1;

    printf("{");
    for (node n = 0; n < N; n++)
        if (set_in(s, n))
        {
            printf("%d, ", n);
            if (empty)
                empty = 0;
        }
    if (!empty)
        printf("\b\b");
    printf("}\n");
}

void set_insert(set* s, node n)
{
    s->flags[n/8] = (unsigned int)s->flags[n/8] | (unsigned int)pow(2, n % 8);
}

void set_difference(set* res, set* s1, set* s2)
{
    for (node n = 0; n < N; n++)
        if (set_in(s1, n) && !set_in(s2, n))
            set_insert(res, n);
}

void set_clear(set* s)
{
    memset(s->flags, 0, DIM_FLAGS*sizeof(char));
}

void set_free(set* s)
{
    free(s);
}

int main(void)
{
    printf("%d\n", sizeof(set));

    set* s1 = set_create();
    set* s2 = set_create();
    set* res = set_create();
    
    set_insert(s1, 0);
    set_clear(s1);
    set_insert(s1, 1);
    set_print(s1);

    set_insert(s2, 1);
    set_insert(s2, 2);
    set_print(s2);

    set_difference(res, s1, s2);
    
    set_print(res);

    set_free(s1);
    set_free(s2);
    set_free(res);

    return 0;
}
