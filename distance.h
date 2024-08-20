
#include "semiring.h"

using namespace std;

class Distance : public Weighable<Distance>
{
    private:
        double d;

    public:
        static Distance zero();
        static Distance unity();
        Distance(double d);
        Distance operator + (const Distance& other) const override;
        Distance operator * (const Distance& other) const override;
        bool operator < (const Distance& other) const override;
        bool operator == (const Distance& other) const override;
        friend ostream& operator << (ostream& os, const Distance& dis);
        ~Distance() = default;
};
