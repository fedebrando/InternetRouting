
#include <vector>
#include <set>
#include <limits>
#include <algorithm>
#include <iterator>
#include "path.h"
#include "semiring.h"
#include "distance.h"
#include <omp.h>
#include <sstream>

using namespace std;

set<node> operator - (const set<node>& s1, const set<node>& s2)
{
    set<node> difference;

    set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), inserter(difference, difference.begin()));
    return difference;
}

template<Semiring T>
class Routing
{
    private:
        set<node> v;
        vector<vector<T>> a;
        vector<vector<T>> d;
        vector<vector<set<Path>>> pi;

        node find_qk_min(node i, const set<node>& s)
        {
            node qk;
            T qk_min = T::zero();

            for (node j : v - s)
                if (d[i][j] < qk_min)
                {
                    qk = j;
                    qk_min = d[i][j];
                }
            return qk;
        }

        void dijkstra(node i)
        {
            set<node> s;
            node qk;

            // initialization
            for (node q : v)
            {
                d[i][q] = T::zero();
            }
            s.clear();
            d[i][i] = T::unity();
            pi[i][i].insert(Path::eps);

            for (int k = 1; k <= v.size(); k++)
            {
                qk = find_qk_min(i, s);
                s.insert(qk);
                for (node j : v - s)
                {
                    if (d[i][qk] * a[qk][j] == d[i][j])
                        pi[i][j] += pi[i][qk] ^ set<Path>{Path(make_pair(qk, j))};
                    else if (d[i][qk] * a[qk][j] < d[i][j])
                    {
                        d[i][j] = d[i][qk] * a[qk][j];
                        pi[i][j] = pi[i][qk] ^ set<Path>{Path(make_pair(qk, j))};
                    } 
                }
            }
        }

    public:
        Routing(const set<node>& v, const vector<vector<T>>& a) : 
            d(v.size(), vector<T>(v.size(), T::zero())), 
            pi(v.size(), vector<set<Path>>(v.size(), set<Path>()))
        {
            this->v = v;
            this->a = a;
        }

        void compute_seq()
        {
            for (node i : v)
                dijkstra(i);
        }

        void compute_par()
        {
            vector<node> v_vec(v.begin(), v.end());

            #pragma omp parallel for schedule(static)
            for (int j = 0; j < v_vec.size(); j++)
            {
                ostringstream os;
                os << "Node " << v_vec[j] << " computed by " << omp_get_thread_num() << " (tot " << omp_get_num_threads() << ")" << endl;
                cout << os.str();
                dijkstra(v_vec[j]);
            }
        }

        vector<vector<T>> getD() const
        {
            return d;
        }

        vector<vector<set<Path>>> getPi() const
        {
            return pi;
        }

        ~Routing() = default;
};

int main (void) 
{
    set<node> v = {0,1,2,3,4,5}; /*{0,1,2,3,4,5};*/
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

    Routing<Distance> r(v, a);
    r.compute_par();

    for (vector<Distance> vd : r.getD())
    {
        for (Distance w : vd)
        {
            cout << w << " ";
        }
        cout << endl;
    }

    for (vector<set<Path>> vsp : r.getPi())
    {
        for (set<Path> sp : vsp)
        {
            cout << "{";
            for (Path p : sp)
            {
                cout << p << " ";
            }
            cout << "\b} ";
        }
        cout << endl;
    }

    return 0;
}