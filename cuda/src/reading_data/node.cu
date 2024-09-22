
#include "node.cuh"

__host__ void Node_print(const Node* n)
{
    printf("%s", n->label);
}

__host__ double to_radians(double degree) 
{
    return degree * M_PI / 180.0;
}

// Returns the real distance in km between two points on the Earth
__host__ double haversine(const Node* n1, const Node* n2)
{
    const double R = 6371.0; // Earth radius in km
    double lat1_rad, lon1_rad, lat2_rad, lon2_rad, dLat_rad, dLon_rad, a, c;

    lat1_rad = to_radians(n1->latitude);
    lon1_rad = to_radians(n1->longitude);
    lat2_rad = to_radians(n2->latitude);
    lon2_rad = to_radians(n2->longitude);
    dLat_rad = lat2_rad - lat1_rad;
    dLon_rad = lon2_rad - lon1_rad;
    a = sin(dLat_rad / 2) * sin(dLat_rad / 2) + cos(lat1_rad) * cos(lat2_rad) * sin(dLon_rad / 2) * sin(dLon_rad / 2);
    c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return R * c;
}

void print_results(const lex_product* d, pset** pi, const Node* v_info, size_t n)
{
    for (node i = 0; i < n; i++)
    {
        for (node j = 0; j < n; j++)
        {
            Node_print(v_info + i);
            printf(" -> ");
            Node_print(v_info + j);
            printf(" ");
            lex_product_print(d[idx(i, j, n)]);
            printf(": ");
            pset_v_info_print(pi[idx(i, j, n)], v_info);
            printf("\n");
        }
    }
}

void pset_v_info_print(const pset* s, const Node* v_info)
{
    printf("{");
    for (node i = 0; i < s->size; i++)
    {
        path_v_info_print(s->paths[i], v_info);
        if (i != s->size - 1)
            printf(", ");
    }
    printf("}");
}

void path_v_info_print(const path* p, const Node* v_info)
{
    if (p->size)
    {
        printf("(");
        for (node i = 0; i < p->size; i++)
        {
            Node_print(v_info + p->nodes[i]);
            if (i == p->size - 1)
                printf(")");
            else
                printf(", ");
        }
    }
    else
        printf("ε");
}
