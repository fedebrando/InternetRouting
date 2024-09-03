
/*
 * Routing algorithm according to a generic semiring metric
*/

#ifndef ROUTING
#define ROUTING

#include <vector>
#include <set>
#include <sstream> // for parallel info printing
#include "path.hpp"
#include "semiring.hpp"
#include "utilities.hpp"
#include "settings.h"

#ifdef PAR_OMP
#include <omp.h>
#endif

using namespace std;

template<Semiring E, typename T>
class Routing
{
    private:
        vector<T> v_info;
        set<node> v;
        vector<vector<E>> a;
        vector<vector<E>> d;
        vector<vector<set<Path<node>>>> pi;

        node find_qk_min(node i, const set<node>& s)
        {
            node qk;
            E qk_min = E::zero();

            for (node j : v - s)
                if (d[i][j] <= qk_min)
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
                d[i][q] = E::zero();
            s.clear();
            d[i][i] = E::unity();
            pi[i][i].insert(Path<node>::eps);

            // dijkstra algorithm
            for (int k = 1; k <= v.size(); k++)
            {
                qk = find_qk_min(i, s);
                s.insert(qk);
                for (node j : v - s)
                {
                    if (d[i][qk] * a[qk][j] == d[i][j])
                        pi[i][j] += pi[i][qk] ^ set<Path<node>>{Path(Edge(qk, j))};
                    else if (d[i][qk] * a[qk][j] < d[i][j])
                    {
                        d[i][j] = d[i][qk] * a[qk][j];
                        pi[i][j] = pi[i][qk] ^ set<Path<node>>{Path(Edge(qk, j))};
                    } 
                }
            }
        }

    public:
        Routing(const vector<T>& v_info, const vector<vector<E>>& a) : 
            d(v_info.size(), vector<E>(v_info.size(), E::zero())),
            pi(v_info.size(), vector<set<Path<node>>>(v_info.size(), set<Path<node>>()))
        {
            this->v_info = v_info;
            this->a = a;
            for (node i = 0; i < v_info.size(); i++)
                v.insert(i);
        }

        Routing(const Routing& r) = delete;

        Routing& operator = (const Routing& r) = delete;

#ifdef SEQ
        void compute()
        {
            for (node i : v)
                dijkstra(i);
        }
#endif

#ifdef PAR_OMP
        void compute()
        {
            vector<node> v_vec(v.begin(), v.end());

            omp_set_nested(1);

            #pragma omp parallel for schedule(static)
            for (int j = 0; j < v_vec.size(); j++)
            {
                /*
                ostringstream os;
                os << "Node " << v_vec[j] << " computed by " << omp_get_thread_num() << " (tot " << omp_get_num_threads() << ")" << endl;
                cout << os.str();
                */
                dijkstra(v_vec[j]);
            }
        }
#endif

        vector<vector<E>> getD() const
        {
            return d;
        }

        vector<vector<set<Path<T>>>> getPi() const
        {
            vector<vector<set<Path<T>>>> pi_info(pi.size(), vector<set<Path<T>>>(pi.size()));
            set<Path<T>> pi_info_el;
            Path<T> el_path_info;

            for (node i = 0; i < pi.size(); i++)
                for (node j = 0; j < pi.size(); j++)
                {
                    for (const Path<node>& el_path : pi[i][j])
                    {
                        for (const Edge<node>& edge : el_path)
                            el_path_info.insert(Edge(v_info[edge.first], v_info[edge.second]));
                        pi_info_el.insert(el_path_info);
                        el_path_info.clear();
                    }
                    pi_info[i][j] = pi_info_el;
                    pi_info_el.clear();
                }
            return pi_info;
        }

        ~Routing() = default;
};

#endif
