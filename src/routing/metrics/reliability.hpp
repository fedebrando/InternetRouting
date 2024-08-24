
/*
 * Most Reliable Path
*/

#ifndef RELIABILITY
#define RELIABILITY

#include <iostream>
#include "semiring.hpp"

using namespace std;

class Reliability : public Weighable<Reliability>
{
    private:
        double r;


    public:
        static Reliability zero();
        static Reliability unity();
        Reliability();
        Reliability(const Reliability& rel);
        Reliability(double r);
        double getR() const;
        void setR(double r);
        Reliability operator + (const Reliability& other) const override;
        Reliability operator * (const Reliability& other) const override;
        bool operator == (const Reliability& other) const override;
        friend ostream& operator << (ostream& os, const Reliability& rel);
        ~Reliability() = default;
};

#endif
