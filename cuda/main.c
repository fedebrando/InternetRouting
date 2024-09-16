
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define MAX_LINE_LENGTH 1024
#define MAX_STRING_LENGTH 25
#define ERR -1

typedef struct 
{
    char country[MAX_STRING_LENGTH];
    char label[MAX_STRING_LENGTH];
    char type[MAX_STRING_LENGTH];
    double latitude;
    double longitude;
} Node;

void print_Node(Node n)
{
    printf("%s", n.label);
}

int get_num_nodes(char* filename)
{
    FILE* file = fopen(filename, "r");
    int n_nodes = 0;
    char line[MAX_LINE_LENGTH];

    if (file == NULL) 
    {
        perror("Error in opening file!");
        return ERR;
    }
    fgets(line, sizeof(line), file);
    while(fgets(line, sizeof(line), file))
        n_nodes++;
    fclose(file);
    return n_nodes;
}

int get_num_values(const char* line)
{
    int count = 1;  // Inizia da 1 perché il numero di valori è numero di virgole + 1
    const char *ptr = line;

    while (*ptr != '\0') {
        if (*ptr == ',') {
            count++;
        }
        ptr++;
    }
    return count;
}

int getV(Node* v, char* filename)
{
    FILE* file = fopen(filename, "r");
    int n_values;
    char** values;
    char line[MAX_LINE_LENGTH];

    if (file == NULL) 
    {
        perror("Error in opening file!");
        return ERR;
    }

    fgets(line, sizeof(line), file); //skip first line
    n_values = get_num_values(line);
    values = (char **)malloc(sizeof(char*) * n_values);
    for(int i = 0; i < n_values; i++)
        values[i] = (char*)malloc(sizeof(char) * MAX_STRING_LENGTH);
    while (fscanf(file, "%[^,],%[^,],%[^,],%[^,],%[^,],%[^\n]\n", 
        values[0], values[1], values[2], values[3], values[4], values[5]) == 6) 
    {
        strcpy(v[atoi(values[0])].country, values[2]);
        strcpy(v[atoi(values[0])].label, values[5]);
        strcpy(v[atoi(values[0])].type, values[3]);
        v[atoi(values[0])].latitude = atof(values[1]);
        v[atoi(values[0])].longitude = atof(values[4]);
    }
    fclose(file);
    return 0;
}

double to_radians(double degree) 
{
    return degree * M_PI / 180.0;
}

// Returns the real distance in km between two points on the Earth
double haversine(Node n1, Node n2)
{
    const double R = 6371.0; // Earth radius in km
    double lat1_rad, lon1_rad, lat2_rad, lon2_rad, dLat_rad, dLon_rad, a, c;

    lat1_rad = to_radians(n1.latitude);
    lon1_rad = to_radians(n1.longitude);
    lat2_rad = to_radians(n2.latitude);
    lon2_rad = to_radians(n2.longitude);
    dLat_rad = lat2_rad - lat1_rad;
    dLon_rad = lon2_rad - lon1_rad;
    a = sin(dLat_rad / 2) * sin(dLat_rad / 2) + cos(lat1_rad) * cos(lat2_rad) * sin(dLon_rad / 2) * sin(dLon_rad / 2);
    c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return R * c;
}

int getA(char* filename, Node* v, double*** a)
{
    FILE *file = fopen(filename, "r");
    int n_values;
    char** values;
    char line[MAX_LINE_LENGTH];
    int first, second;
    double distance;

    if (file == NULL) 
    {
        perror("Error in opening file!");
        return ERR;
    }

    fgets(line, sizeof(line), file);
    n_values = get_num_values(line);
    values = (char **)malloc(sizeof(char*) * n_values);
    for(int i = 0; i < n_values; i++)
        values[i] = (char*)malloc(sizeof(char) * MAX_STRING_LENGTH);
    while(fscanf(file, "%[^,],%[^,],%[^,],%[^\n]\n", 
        values[0], values[1], values[2], values[3]) == 4)
    {
        first = atoi(values[0]);
        second = atoi(values[1]);
        distance = haversine(v[first], v[second]);
        a[first][second][0] = distance;
        a[second][first][0] = distance;
    }

    fclose(file);
    return 0;
}

int main(void)
{
    int n_nodes;
    Node* v;
    double*** a;
    
    n_nodes = get_num_nodes("node.dat");
    if (n_nodes == ERR)
        exit(ERR);
    v = (Node*)malloc(sizeof(Node) * n_nodes);
    if(getV(v, "node.dat") == ERR)
        exit(ERR);
    
    /* stampa dei nodi
    for(int i = 0; i < n_nodes; i++)
    {
        print_Node(nodes[i]);
        printf("\n");
    }
    */

    a = (double***) malloc(sizeof(double**) * n_nodes);
    for (int i = 0; i < n_nodes; i++)
        a[i] = (double**) malloc(sizeof(double*) * n_nodes);
    
    for (int i = 0; i < n_nodes; i++)
        for (int j = 0; j < n_nodes; j++)
            a[i][j] = (double*) malloc(sizeof(double) * 2);

    for (int i = 0; i < n_nodes; i++)
        for (int j = 0; j < n_nodes; j++)
            a[i][j][0] = INFINITY;

    

    if (getA("edge.dat", v, a) == -1)
        exit(ERR);
    
    
    for (int i = 0; i < n_nodes; i++)
    {
        for (int j = 0; j < n_nodes; j++)
            printf("%lf ", a[i][j][0]);
        printf("\n");
    }
    

    return 0;
}


// array di struct Node -> v
// matrice di double[2] -> a (double** a[2])
// rifare tutto come in c++
// a host->device
