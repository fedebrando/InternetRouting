
#ifndef PATH
#define PATH

#include <vector>
#include <iostream>
#include "edge.hpp"
#include "utilities.hpp"
#include "settings.h"

#ifdef PAR_OMP
#include <omp.h>
#endif

using namespace std;

/*
 * Class Path<T> HAS-A Edge<T> 
*/
template <Order T>
class Path
{
    private:
        vector<Edge<T>> path;
        bool loop_free;
        set<T> nodes;

        bool in(const vector<T>& v, T n)
        {
            for (T nn : v)
                if (nn == n)
                    return true;
            return false;
        }

    public:
        static const Path<T> eps;

        Path() : path(), loop_free(true), nodes()
        {}

        Path(const Path& p) : nodes(p.nodes)
        {
            path = vector<Edge<T>>(p.path);
            loop_free = p.is_loop_free();
        }

        Path(Edge<T> e)
        {
            path.push_back(e);
            loop_free = true;
            nodes.insert(e.first);
            nodes.insert(e.second);
        }

        vector<Edge<T>>::iterator begin()
        {
            return path.begin();
        }

        vector<Edge<T>>::iterator end()
        {
            return path.end();
        }

        vector<Edge<T>>::const_iterator begin() const
        {
            return path.begin();
        }

        vector<Edge<T>>::const_iterator end() const
        {
            return path.end();
        }

        bool is_loop_free() const
        {
            return loop_free;
        }

        void insert(Edge<T> e) //O(n)
        {
            if (!path.empty())
                if (!(path.back().second == e.first))
                    return;
            path.push_back(e);
            if (is_loop_free() && nodes.find(e.second) != nodes.end())
                loop_free = false;
            nodes.insert(e.second);
        }

        void clear()
        {
            path.clear();
        }

        Path operator + (const Path& other) const //O(n)
        {
            Path p;

            if (!(path.empty() || other.path.empty()))
                if (!(path.back().second == other.path.front().first))
                    throw runtime_error("Cannot concat non contiguous paths!");

            p.path.insert(p.path.end(), path.begin(), path.end());
            for (const Edge<T>& e : other)
                p.insert(e);
            return p;
        }

        bool operator < (const Path& other) const //necessary for set
        {
            return path.size() < other.path.size();
            //return path < other.path;
        }

        bool operator == (const Path& other) const
        {
            return path == other.path;
        }

        friend ostream& operator << (ostream& os, const Path& p)
        {
            if (p == eps)
                os << "ε";
            else
            {
                os << "(" << p.path[0].first;
                for (const Edge<T>& e : p.path)
                    os << ", " << e.second;
                os << ")";
            }
            return os;
        }

        Path<T>& operator = (const Path<T>& p)
        {
            path = p.path;
            return *this;
        }

        ~Path() = default;
};

template <Order T>
const Path<T> Path<T>::eps;

#ifdef SEQ
template <Order T>
set<Path<T>> operator ^ (const set<Path<T>>& s1, const set<Path<T>>& s2) //O(n^2)
{
    set<Path<T>> res;
    Path<T> p;

    for (const Path<T>& p1 : s1)
        for (const Path<T>& p2 : s2)
        {
            p = p1 + p2;
            if (p.is_loop_free())
                res.insert(p);
        }
    return res;
}
#endif
#ifdef PAR_OMP
template <Order T>
set<Path<T>> operator ^ (const set<Path<T>>& s1, const set<Path<T>>& s2) //O(n^2)
{
    set<Path<T>> res;
    Path<T> p;
    vector<Path<T>> v1(s1.begin(), s1.end());

    #pragma omp parallel for schedule(dynamic) shared(res)
    for (int i = 0; i < v1.size(); i++)
    {
        
        ostringstream os;
        os << "Path computed by " << omp_get_thread_num() << " (tot " << omp_get_num_threads() << ")" << endl;
        cout << os.str();
        
        for (const Path<T>& p2 : s2)
        {
            p = v1[i] + p2;
            if (p.is_loop_free())
            {
                #pragma omp critical
                res.insert(p);
            }
        }
    }
    return res;
}
#endif
#ifdef PAR_CUDA
#endif

template <Order T>
void operator += (set<Path<T>>& s1, const set<Path<T>>& s2)
{
    s1.insert(s2.begin(), s2.end());
}

#endif
