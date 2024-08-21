
#include "routing.hpp"
#include <string>

using namespace std;

int main (void) 
{
    vector<string> v = {"A", "B", "C", "D", "E", "F"};
    vector<vector<Distance>> a =
    /*
    {
        {Distance::zero(), Distance(1), Distance::zero(), Distance::zero(), Distance::zero()},
        {Distance(1), Distance::zero(), Distance(2), Distance(1), Distance(1)},
        {Distance::zero(), Distance(2), Distance::zero(), Distance(1), Distance(1)},
        {Distance::zero(), Distance(1), Distance(1), Distance::zero(), Distance::zero()},
        {Distance::zero(), Distance(1), Distance(1), Distance::zero(), Distance::zero()}
    };
    */
    {
        {Distance::zero(), Distance(1), Distance::zero(), Distance::zero(), Distance::zero(), Distance(3)},
        {Distance(1), Distance::zero(), Distance(3), Distance::zero(), Distance(5), Distance(1)},
        {Distance::zero(), Distance(3), Distance::zero(), Distance(2), Distance::zero(), Distance::zero()},
        {Distance::zero(), Distance::zero(), Distance(2), Distance::zero(), Distance(1), Distance(6)},
        {Distance::zero(), Distance(5), Distance::zero(), Distance(1), Distance::zero(), Distance(2)},
        {Distance(3), Distance(1), Distance::zero(), Distance(6), Distance(2), Distance::zero()}
    };

    Routing<Distance, string> r(v, a);
    r.compute_par();

    for (vector<Distance> vd : r.getD())
    {
        for (Distance w : vd)
        {
            cout << w << " ";
        }
        cout << endl;
    }

    for (vector<set<Path<string>>> vsp : r.getPi())
    {
        for (set<Path<string>> sp : vsp)
        {
            cout << "{";
            for (const Path<string>& p : sp)
            {
                cout << p << " ";
            }
            cout << "\b} ";
        }
        cout << endl;
    }

    return 0;
}