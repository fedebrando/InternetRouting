
#include <vector>
#include <set>
#include <limits>
#include <algorithm>
#include <iterator>
#include "path.h"

using namespace std;

set<node> operator - (const set<node>& s1, const set<node>& s2)
{
    set<node> difference;

    set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), inserter(difference, difference.begin()));
    return difference;
}

set<Path> operator ^ (const set<Path>& s1, const set<Path>& s2)
{
    set<Path> res;
    Path p;

    for (const Path& p1 : s1)
        for (const Path& p2 : s2)
        {
            p = p1 + p2;
            if (p.loop_free())
                res.insert(p);
        }
    return res;
}

void operator += (set<Path>& s1, const set<Path>& s2)
{
    s1.insert(s2.begin(), s2.end());
}

int main (void) 
{
    set<node> v = {0,1,2,3,4,5};
    double zero = numeric_limits<double>::infinity();
    double unity = 0;
    vector<vector<double>> a =
    {
        {zero, 1, zero, zero, zero, 3},
        {1, zero, 3, zero, 5, 1},
        {zero, 3, zero, 2, zero, zero},
        {zero, zero, 2, zero, 1, 6},
        {zero, 5, zero, 1, zero, 2},
        {3, 1, zero, 6, 2, zero}
    };
    Path eps;
    vector<vector<double>> d(6, vector<double>(6, zero));
    vector<vector<set<Path>>> pi(6, vector<set<Path>>(6, set<Path>()));
    set<node> s;
    node qk;
    double qk_min;
    
    for (node i : v)
    {
        for (node q : v)
        {
            d[i][q] = zero;
        }
        s.clear();
        d[i][i] = unity;
        pi[i][i].insert(eps);
        
        for (int k = 1; k <= v.size(); k++)
        {
            
            qk_min = zero;
            for (node j : v - s)
            {
                if (d[i][j] < qk_min)
                {
                    qk = j;
                    qk_min = d[i][j];
                }
            }
            s.insert(qk);
            
            for (node j : v - s)
            {
                
                if (d[i][qk] + a[qk][j] == d[i][j])
                    pi[i][j] += set<Path>{Path(make_pair(qk, j))};
                else if (d[i][qk] + a[qk][j] < d[i][j])
                {
                    d[i][j] = d[i][qk] + a[qk][j];
                    pi[i][j] = pi[i][qk] ^ set<Path>{Path(make_pair(qk, j))};
                } 
            }
        }
    }
    
    

    for (vector<double> vd : d)
    {
        for (double w : vd)
        {
            cout << w << " ";
        }
        cout << endl;
    }

    for (vector<set<Path>> vsp : pi)
    {
        for (set<Path> sp : vsp)
        {
            for (Path p : sp)
            {
                cout << p << " ";
            }
        }
        cout << endl;
    }

    return 0;
}