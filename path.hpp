
#ifndef PATH
#define PATH

#include <vector>
#include <utility>
#include <iostream>
#include <concepts>

using namespace std;

typedef unsigned int node;

template<typename T>
concept Order = requires(T a, T b)
{
    {a == b} -> convertible_to<bool>;
    {a < b} -> convertible_to<bool>;
};

template<typename T>
concept Print = requires(ostream& os, T a)
{
    { os << a } -> convertible_to<ostream&>;
};

template <typename T>
requires Order<T> && Print<T>
class Edge : public pair<T, T>
{
    public:
        Edge() : pair<T, T>()
        {}

        Edge(T first, T second) : pair<T, T>(first, second)
        {}
};

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
            for (const Edge<T>& e : p.path)
                os << "(" << e.first << ", " << e.second << ")";
            return os;
        }

        ~Path() = default;
};

template <typename T>
const Path<T> Path<T>::eps = Path<T>();

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
