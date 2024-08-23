
#include "reliability.hpp"

Reliability Reliability::zero()
{
    static Reliability r0(0);

    return r0;
}

Reliability Reliability::unity()
{
    static Reliability r1(1);

    return r1;
}

Reliability::Reliability() : Reliability(Reliability::zero())
{}

Reliability::Reliability(const Reliability& rel)
{
    this->r = rel.r;
}

Reliability::Reliability(double r)
{
    this->r = r;
}

Reliability Reliability::operator + (const Reliability& other) const 
{
    return Reliability(max(r, other.r));
}

Reliability Reliability::operator * (const Reliability& other) const 
{
    return Reliability(r * other.r);
}

bool Reliability::operator == (const Reliability& other) const 
{
    return r == other.r;
}

ostream& operator << (ostream& os, const Reliability& rel)
{
    os << rel.r;
    return os;
}