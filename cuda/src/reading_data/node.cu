
#include "node.cuh"

__host__ void Node_print(Node n)
{
    printf("%s", n.label);
}

__host__ double to_radians(double degree) 
{
    return degree * M_PI / 180.0;
}

// Returns the real distance in km between two points on the Earth
__host__ double haversine(Node n1, Node n2)
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
