
#include "bandwidth.hpp"

Bandwidth Bandwidth::zero()
{
    static Bandwidth bw0(0);

    return bw0;
}

Bandwidth Bandwidth::unity()
{
    static Bandwidth bw1(numeric_limits<double>::infinity());

    return bw1;
}


Bandwidth::Bandwidth() : Bandwidth(Bandwidth::zero())
{}

Bandwidth::Bandwidth(const Bandwidth& bandw)
{
    this->bw = bandw.bw;
}

Bandwidth::Bandwidth(double bw)
{
    this->bw = bw;
}

Bandwidth Bandwidth::operator + (const Bandwidth& other) const 
{
    return Bandwidth(max(bw, other.bw));
}

Bandwidth Bandwidth::operator * (const Bandwidth& other) const 
{
    return Bandwidth(min(bw, other.bw));
}

bool Bandwidth::operator == (const Bandwidth& other) const 
{
    return bw == other.bw;
}

ostream& operator << (ostream& os, const Bandwidth& bandw)
{
    os << bandw.bw;
    return os;
}