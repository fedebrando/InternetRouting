/*
 * Widest Path
*/

#ifndef BANDWIDTH
#define BANDWIDTH

#include <limits>
#include <iostream>
#include "semiring.hpp"

using namespace std;

class Bandwidth : public Weighable<Bandwidth>
{
    private:
        double bw;

    public:
        static Bandwidth zero();
        static Bandwidth unity();
        Bandwidth();
        Bandwidth(const Bandwidth& bandw);
        Bandwidth(double bw);
        double getBw() const;
        void setBw(double bw);
        Bandwidth operator + (const Bandwidth& other) const override;
        Bandwidth operator * (const Bandwidth& other) const override;
        bool operator == (const Bandwidth& other) const override;
        friend ostream& operator << (ostream& os, const Bandwidth& bandw);
        ~Bandwidth() = default;
};

#endif
