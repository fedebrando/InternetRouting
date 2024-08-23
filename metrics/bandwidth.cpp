
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

Bandwidth::Bandwidth(const Bandwidth& bandw) : Bandwidth(bandw.getBw())
{}

Bandwidth::Bandwidth(double bw)
{
    setBw(bw);
}

double Bandwidth::getBw() const
{
    return bw;
}

void Bandwidth::setBw(double bw)
{
    if (bw >= 0)
        this->bw = bw;
    else
        this->bw = zero().getBw();
}

Bandwidth Bandwidth::operator + (const Bandwidth& other) const 
{
    return Bandwidth(max(getBw(), other.getBw()));
}

Bandwidth Bandwidth::operator * (const Bandwidth& other) const 
{
    return Bandwidth(min(getBw(), other.getBw()));
}

bool Bandwidth::operator == (const Bandwidth& other) const 
{
    return getBw() == other.getBw();
}

ostream& operator << (ostream& os, const Bandwidth& bandw)
{
    os << bandw.getBw();
    return os;
}