
#include <limits>
#include "iweight.h"

using namespace std;

class Distance : public IWeight
{
    private:
        double distance;
    public:
        Distance(double distance = 0)
        {
            this->distance = distance;
        }
};