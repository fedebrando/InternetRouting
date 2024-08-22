
#include "distance.hpp"

const Distance Distance::zero = Distance(numeric_limits<double>::infinity());

const Distance Distance::unity = Distance(0);

Distance::Distance() : Distance(Distance::zero)
{}

Distance::Distance(const Distance& dis)
{
    this->d = dis.d;
}

Distance::Distance(double d)
{
    this->d = d;
}

Distance Distance::operator + (const Distance& other) const 
{
    return Distance(min(d, other.d));
}

Distance Distance::operator * (const Distance& other) const 
{
    return Distance(d + other.d);
}

bool Distance::operator == (const Distance& other) const 
{
    return d == other.d;
}

ostream& operator << (ostream& os, const Distance& dis)
{
    os << dis.d;
    return os;
}