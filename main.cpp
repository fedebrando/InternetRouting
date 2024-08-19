#include <iostream>
#include <limits>
#include "semiring.h"

using namespace std;

double min(double a, double b)
{
    return (a <= b) ? a : b;
}

double sum(double a, double b)
{
    return a + b;
}

int main (void)
{
    double zero = numeric_limits<double>::infinity();
    double unity = 0;
    Semiring<double> s(min, sum, zero, unity);

    cout << s.+(2,3) << endl;

    return 0;
}