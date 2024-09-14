
#include "node.hpp"

Node::Node() :  Node("", "", "", 0, 0)
{}

Node::Node(string country, string label, string type, double latitude, double longitude)
{
    setCountry(country);
    setLabel(label);
    setType(type);
    setLatitude(latitude);
    setLongitude(longitude);
}

string Node::getCountry() const
{
    return country;
}

string Node::getLabel() const
{
    return label;
}

string Node::getType() const
{
    return type;
}

double Node::getLatitude() const
{
    return latitude;
}

double Node::getLongitude() const
{
    return longitude;
}

void Node::setCountry(string country)
{
    this->country = country;
}

void Node::setLabel(string label)
{
    this->label = label;
}

void Node::setType(string type)
{
    this->type = type;
}

void Node::setLatitude(double latitude)
{
    if (-90 <= latitude && latitude <= 90)
        this->latitude = latitude;
    else
        throw invalid_argument("Invalid latitude");
}

void Node::setLongitude(double longitude)
{
    if (-180 <= longitude && longitude <= 180)
        this->longitude = longitude;
    else
        throw invalid_argument("Invalid longitude");
}

bool Node::operator == (const Node& n) const
{
    return getCountry() == n.getCountry() && getLabel() == n.getLabel() && getType() == n.getType() && getLatitude() == n.getLatitude() && getLongitude() == n.getLongitude(); 
}

bool Node::operator < (const Node& n) const
{
    return getLabel() < n.getLabel();
}

ostream& operator << (ostream& os, const Node& n)
{
    os << n.getLabel();
    return os;
}

double to_radians(double degree) 
{
    return degree * M_PI / 180.0;
}

// Returns the real distance in km between two points on the Earth
double haversine(const Node& n1, const Node& n2)
{
    const double R = 6371.0; // Earth radius in km
    double lat1_rad, lon1_rad, lat2_rad, lon2_rad, dLat_rad, dLon_rad, a, c;

    lat1_rad = to_radians(n1.getLatitude());
    lon1_rad = to_radians(n1.getLongitude());
    lat2_rad = to_radians(n2.getLatitude());
    lon2_rad = to_radians(n2.getLongitude());
    dLat_rad = lat2_rad - lat1_rad;
    dLon_rad = lon2_rad - lon1_rad;
    a = sin(dLat_rad / 2) * sin(dLat_rad / 2) + cos(lat1_rad) * cos(lat2_rad) * sin(dLon_rad / 2) * sin(dLon_rad / 2);
    c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return R * c;
}
