
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

int main(void)
{
    int n_nodes;
    Node* nodes;
    
    n_nodes = get_num_nodes("node.dat");
    if (n_nodes == ERR)
        exit(ERR);
    nodes = (Node*)malloc(sizeof(Node) * n_nodes);
    if(getV(nodes, "node.dat") == ERR)
        exit(ERR);
    for(int i = 0; i < n_nodes; i++)
    {
        print_Node(nodes[i]);
        printf("\n");
    }

    return 0;
}


// array di struct Node -> v
// matrice di double[2] -> a (double** a[2])
// rifare tutto come in c++
// a host->device
