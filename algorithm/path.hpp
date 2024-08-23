
#ifndef PATH
#define PATH

#include <vector>
#include <utility>
#include <iostream>
#include "utilities.hpp"

using namespace std;

template <typename T>
class Path
{
    private:
        vector<Edge<T>> path;

        bool in(const vector<T>& v, T n)
        {
            for (T nn : v)
                if (nn == n)
                    return true;
            return false;
        }

    public:
        static const Path<T> eps;

        Path() : path()
        {}

        Path(const Path& p)
        {
            path = vector<Edge<T>>(p.path);
        }

        Path(Edge<T> e)
        {
            path.push_back(e);
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

        bool loop_free()
        {
            vector<T> v;

            if (path.empty())
                return true;
            v.push_back(path[0].first);
            for (const Edge<T>& e : path)
            {
                if (in(v, e.second))
                    return false;
                v.push_back(e.second);
            }
            return true;
        }

        void insert(Edge<T> e)
        {
            if (!path.empty())
                if (!(path.back().second == e.first))
                    return;
            path.push_back(e);
        }

        void clear()
        {
            path.clear();
        }

        Path operator + (const Path& other) const
        {
            Path p;

            if (!(path.empty() || other.path.empty()))
                if (!(path.back().second == other.path.front().first))
                    throw runtime_error("Cannot concat non contiguous paths!");

            p.path.insert(p.path.end(), path.begin(), path.end());
            p.path.insert(p.path.end(), other.path.begin(), other.path.end());
            return p;
        }

        bool operator < (const Path& other) const //necessary for set
        {
            return path < other.path;
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

template <typename T>
const Path<T> Path<T>::eps;

template <typename T>
set<Path<T>> operator ^ (const set<Path<T>>& s1, const set<Path<T>>& s2)
{
    set<Path<T>> res;
    Path<T> p;

    for (const Path<T>& p1 : s1)
        for (const Path<T>& p2 : s2)
        {
            p = p1 + p2;
            if (p.loop_free())
                res.insert(p);
        }
    return res;
}

template <typename T>
void operator += (set<Path<T>>& s1, const set<Path<T>>& s2)
{
    s1.insert(s2.begin(), s2.end());
}

#endif
