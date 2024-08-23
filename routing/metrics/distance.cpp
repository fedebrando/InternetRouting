
#include "distance.hpp"

Distance Distance::zero()
{
    static Distance d0(numeric_limits<double>::infinity());

    return d0;
}

Distance Distance::unity()
{
    static Distance d1(0);

    return d1;
}

Distance::Distance() : Distance(Distance::zero())
{}

Distance::Distance(const Distance& dis) : Distance(dis.getD())
{}

Distance::Distance(double d)
{
    setD(d);
}

double Distance::getD() const
{
    return d;
}

void Distance::setD(double d)
{
    if (d >= 0)
        this->d = d;
    else
        this->d = zero().getD();
}

Distance Distance::operator + (const Distance& other) const 
{
    return Distance(min(getD(), other.getD()));
}

Distance Distance::operator * (const Distance& other) const 
{
    return Distance(getD() + other.getD());
}

bool Distance::operator == (const Distance& other) const 
{
    return getD() == other.getD();
}

ostream& operator << (ostream& os, const Distance& dis)
{
    os << dis.getD();
    return os;
}
