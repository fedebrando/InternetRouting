
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

Reliability::Reliability(const Reliability& rel) : Reliability(rel.getR())
{}

Reliability::Reliability(double r)
{
    setR(r);
}

double Reliability::getR() const
{
    return r;
}

void Reliability::setR(double r)
{
    if (0 <= r && r <= 1)
        this->r = r;
    else
        this->r = zero().getR();
}

Reliability Reliability::operator + (const Reliability& other) const 
{
    return Reliability(max(getR(), other.getR()));
}

Reliability Reliability::operator * (const Reliability& other) const 
{
    return Reliability(getR() * other.getR());
}

bool Reliability::operator == (const Reliability& other) const 
{
    return getR() == other.getR();
}

ostream& operator << (ostream& os, const Reliability& rel)
{
    os << rel.getR();
    return os;
}
