/*
 * Shortest Path
*/

#ifndef DISTANCE
#define DISTANCE

#include <limits>
#include <iostream>
#include "semiring.hpp"

using namespace std;

class Distance : public Weighable<Distance>
{
    private:
        double d;

    public:
        static const Distance zero;
        static const Distance unity;
        Distance();
        Distance(const Distance& dis);
        Distance(double d);
        Distance operator + (const Distance& other) const override;
        Distance operator * (const Distance& other) const override;
        bool operator == (const Distance& other) const override;
        friend ostream& operator << (ostream& os, const Distance& dis);
        ~Distance() = default;
};

#endif
