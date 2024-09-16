
#include <stdio.h>
#include <stdlib.h>

#define N 5
#define PROB_MAX_OPT_PATHS 3

typedef unsigned int node;

typedef struct
{
    int size;
    node nodes[N];
} path;

path* path_create()
{
    path* p = (path*) malloc(sizeof(path));

    p->size = 0;
    return p;
}

void path_free(path* p)
{
    free(p);
}

int path_in(const path* p, node n)
{
    for (int i = 0; i < p->size; i++)
        if (p->nodes[i] == n)
            return 1;
    return 0;
}

int path_add(path* p, node i, node j)
{
    if (p->size)
    {
        if (p->nodes[p->size - 1] != i)
            return 0;
        if (path_in(p, j))
            return 0;
        p->nodes[p->size] = j;
        p->size++;
    }
    else
    {
        p->nodes[0] = i;
        p->nodes[1] = j;
        p->size += 2;
    }
    return 1;
}

int path_equal(const path* p1, const path* p2)
{
    if (p1->size != p2->size)
        return 0;
    for (int i = 0; i < p1->size; i++)
        if (p1->nodes[i] != p2->nodes[i])
            return 0;
    return 1;
}

void path_print(const path* p)
{
    if (p->size)
    {
        printf("(");
        for (int i = 0; i < p->size; i++)
            printf("%d, ", p->nodes[i]);
        printf("\b\b)");
    }
    else
        printf("ε");
}

// ********************************************************

typedef struct
{
    int size;
    int m_size;
    path** paths;
} pset;

pset* pset_create()
{
    pset* s = (pset*) malloc(sizeof(pset));

    s->size = 0;
    s->m_size = PROB_MAX_OPT_PATHS;
    s->paths = (path**) malloc(s->m_size * sizeof(path*));
    return s;
}

void pset_free(pset* s)
{
    free(s->paths);
    free(s);
}

int pset_in(const pset* s, const path* p)
{
    for (int i = 0; i < s->size; i++)
        if (path_equal(s->paths[i], p))
            return 1;
    return 0;
}

void pset_insert(pset* s, path* p)
{
    if (s->size < s->m_size)
    {
        if (pset_in(s, p))
            return;
        s->paths[s->size] = p;
        s->size++;
    }
    else
    {
        s->paths = (path**) realloc(s->paths, (s->m_size + 1) * sizeof(path*));
        s->m_size++;
        pset_insert(s, p);
    }
}

void pset_print(const pset* s)
{
    printf("{");
    for (int i = 0; i < s->size; i++)
    {
        path_print(s->paths[i]);
        if (i != s->size - 1)
            printf(", ");
    }
    printf("}");
}

void pset_pair_wise_concat(pset* res, const pset* s, node i, node j, path* paths_app)
{
    for (int p = 0; p < s->size; p++)
    {
        paths_app[p] = *(s->paths[p]);
        if (path_add(paths_app + p, i, j))
            pset_insert(res, paths_app + p);
    }
}

int main(void)
{
    pset* s = pset_create();
    path* p1 = path_create();
    path* p2 = path_create();
    path* p3 = path_create();
    path* paths_app;
    pset* res = pset_create();

    path_add(p1, 0, 1);
    path_add(p1, 1, 2);

    path_add(p2, 3, 2);

    pset_insert(s, p1);
    pset_insert(s, p2);
    pset_insert(s, p3);
    pset_insert(s, p3);
    pset_insert(s, p3);
    pset_print(s);

    paths_app = (path*) malloc(s->size * sizeof(path));
    pset_pair_wise_concat(res, s, 2, 1, paths_app);

    pset_print(res);

    pset_free(s);
    path_free(p1);
    path_free(p2);
    path_free(p3);
    free(paths_app);
    pset_free(res);

    return 0;
}
