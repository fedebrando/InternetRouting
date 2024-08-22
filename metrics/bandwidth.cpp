
#include "bandwidth.hpp"

const Bandwidth Bandwidth::zero = Bandwidth(0);

const Bandwidth Bandwidth::unity = Bandwidth(numeric_limits<double>::infinity());

Bandwidth::Bandwidth() : Bandwidth(Bandwidth::zero)
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