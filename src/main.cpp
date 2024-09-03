
#include <iostream>
#include <vector>
#include "routing.hpp"
#include "metrics.hpp"
#include "lex_product.hpp"
#include "utilities.hpp"

#define METRIC LexProduct<Distance, Bandwidth>

using namespace std;

int main (void) 
{
    vector<string> v = {"0", "1", "2", "3", "4"};
    vector<vector<METRIC>> a(v.size(), vector<METRIC>(v.size()));
    // distance del paper
    /*
    {
        {Distance::zero(), Distance(1), Distance::zero(), Distance::zero(), Distance::zero()},
        {Distance(1), Distance::zero(), Distance(2), Distance(1), Distance(1)},
        {Distance::zero(), Distance(2), Distance::zero(), Distance(1), Distance(1)},
        {Distance::zero(), Distance(1), Distance(1), Distance::zero(), Distance::zero()},
        {Distance::zero(), Distance(1), Distance(1), Distance::zero(), Distance::zero()}
    };
    
    // distance video yt
    {
        {Distance::zero(), Distance(1), Distance::zero(), Distance::zero(), Distance::zero(), Distance(3)},
        {Distance(1), Distance::zero(), Distance(3), Distance::zero(), Distance(5), Distance(1)},
        {Distance::zero(), Distance(3), Distance::zero(), Distance(2), Distance::zero(), Distance::zero()},
        {Distance::zero(), Distance::zero(), Distance(2), Distance::zero(), Distance(1), Distance(6)},
        {Distance::zero(), Distance(5), Distance::zero(), Distance(1), Distance::zero(), Distance(2)},
        {Distance(3), Distance(1), Distance::zero(), Distance(6), Distance(2), Distance::zero()}
    };
    */

    // distance * reliability
    /*
    a[0][1] = METRIC(Distance(1), Reliability(0.5));
    a[1][4] = METRIC(Distance(1), Reliability(0.1));
    a[1][3] = METRIC(Distance(1), Reliability(0.9));
    a[3][2] = METRIC(Distance(1), Reliability(0.9));
    a[4][2] = METRIC(Distance(1), Reliability(1));
    a[1][2] = METRIC(Distance(2), Reliability(0.81));
    a[1][0] = METRIC(Distance(1), Reliability(0.5));
    a[4][1] = METRIC(Distance(1), Reliability(0.1));
    a[3][1] = METRIC(Distance(1), Reliability(0.9));
    a[2][3] = METRIC(Distance(1), Reliability(0.9));
    a[2][4] = METRIC(Distance(1), Reliability(1));
    a[2][1] = METRIC(Distance(2), Reliability(0.81));
    */

    // distance * bandwidth
    a[0][1] = METRIC(Distance(1), Bandwidth(10));
    a[1][0] = METRIC(Distance(1), Bandwidth(10));
    a[1][2] = METRIC(Distance(2), Bandwidth(90));
    a[2][1] = METRIC(Distance(2), Bandwidth(90));
    a[1][3] = METRIC(Distance(1), Bandwidth(5));
    a[3][1] = METRIC(Distance(1), Bandwidth(5));
    a[1][4] = METRIC(Distance(1), Bandwidth(100));
    a[4][1] = METRIC(Distance(1), Bandwidth(100));
    a[3][2] = METRIC(Distance(1), Bandwidth(100));
    a[2][3] = METRIC(Distance(1), Bandwidth(100));
    a[4][2] = METRIC(Distance(1), Bandwidth(100));
    a[2][4] = METRIC(Distance(1), Bandwidth(100));

    // bandwidth * distance
    /*
    a[0][1] = METRIC(Bandwidth(5), Distance(1));
    a[2][1] = METRIC(Bandwidth(5), Distance(4));
    a[3][0] = METRIC(Bandwidth(5), Distance(1));
    a[4][0] = METRIC(Bandwidth(10), Distance(5));
    a[4][2] = METRIC(Bandwidth(5), Distance(1));
    a[2][3] = METRIC(Bandwidth(5), Distance(1));
    a[3][4] = METRIC(Bandwidth(10), Distance(1));
    */
   
    /*
    cout << "(" << Distance::zero() << ")" << endl;
    cout << "[" << Bandwidth::zero() << "]" << endl;
    cout << LexProduct<Distance, Bandwidth>(Distance::zero(), Bandwidth::zero()) << endl;
    cout << METRIC::zero() << endl;
    cout << LexProduct<Distance, Bandwidth>::unity() << endl;
    */

    Routing<METRIC, string> r(v, a);

    r.compute();
    cout << r.getD() << endl;
    cout << r.getPi() << endl;

    return 0;
}
