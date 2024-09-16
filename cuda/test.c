#include <stdio.h>
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

double to_radians(double degree) 
{
    return degree * M_PI / 180.0;
}

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
    double g = dLat_rad / 2;
    a = sin(g) ;
    return R * c;
}

int main(void)
{
    printf("%f", cos(0));
    return 0;
}