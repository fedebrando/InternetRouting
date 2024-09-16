
#include "reading_data.h"

int get_num_nodes(const char* filename)
{
    FILE* file = fopen(filename, "r");
    int n_nodes = 0;
    char line[MAX_LINE_LENGTH];

    if (!file) 
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
    int count = 1;
    const char *ptr = line;

    while (*ptr != '\0') 
    {
        if (*ptr == ',')
            count++;
        ptr++;
    }
    return count;
}

int getV(Node* v, const char* filename)
{
    FILE* file = fopen(filename, "r");
    int n_values;
    char** values;
    char line[MAX_LINE_LENGTH];

    if (!file) 
    {
        perror("Error in opening file!");
        return ERR;
    }

    fgets(line, sizeof(line), file); //skip first line
    n_values = get_num_values(line);
    values = (char **) malloc(sizeof(char*) * n_values);
    for(int i = 0; i < n_values; i++)
        values[i] = (char*) malloc(sizeof(char) * MAX_STRING_LENGTH);
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

int getA(const char* filename, const Node* v, double*** a)
{
    FILE *file = fopen(filename, "r");
    int n_values;
    char** values;
    char line[MAX_LINE_LENGTH];
    int first, second;
    double distance;
#ifdef SHORTEST_WIDEST
    double bandwidth;
#else
    double reliability;
#endif

    if (!file) 
    {
        perror("Error in opening file!");
        return ERR;
    }

    fgets(line, sizeof(line), file);
    n_values = get_num_values(line);
    values = (char **) malloc(sizeof(char*) * n_values);
    for(int i = 0; i < n_values; i++)
        values[i] = (char*) malloc(sizeof(char) * MAX_STRING_LENGTH);
    while(fscanf(file, "%[^,],%[^,],%[^,],%[^\n]\n", 
        values[0], values[1], values[2], values[3]) == 4)
    {
        first = atoi(values[0]);
        second = atoi(values[1]);
        distance = haversine(v[first], v[second]);
        a[first][second][0] = distance;
        a[second][first][0] = distance;
#ifdef SHORTEST_WIDEST
        bandwidth = atof(values[3]);
        a[first][second][1] = bandwidth;
        a[second][first][1] = bandwidth;
#else
        reliability = atof(values[2]);
        a[first][second][1] = reliability;
        a[second][first][1] = reliability;
#endif
    }

    fclose(file);
    return 0;
}
