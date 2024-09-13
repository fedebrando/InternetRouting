
#ifndef NODE
#define NODE

#include <iostream>
#include <cmath>

using namespace std;

class Node
{
    private:
        string country;
        string label;
        string type;
        double latitude;
        double longitude;

    public:
        Node();
        Node(string country, string label, string type, double latitude, double longitude);
        string getCountry() const;
        string getLabel() const;
        string getType() const;
        double getLatitude() const;
        double getLongitude() const;
        void setCountry(string country);
        void setLabel(string label);
        void setType(string type);
        void setLatitude(double latitude);
        void setLongitude(double longitude);
        bool operator == (const Node& n) const;
        bool operator < (const Node& n) const;
        friend ostream& operator << (ostream& os, const Node& n);
        ~Node() = default;
};

double to_radians(double degree);
double haversine(const Node& n1, const Node& n2);

#endif
