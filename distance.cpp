
#include <limits>
#include <iostream>
#include "distance.h"

static Distance zero()
{
    return Distance(numeric_limits<double>::infinity());
}

static Distance unity()
{
    return Distance(0);
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