
/*
 * Utilities used by Path and Routing
*/

#ifndef UTILITIES
#define UTILITIES

#include <iostream>
#include <set>
#include <concepts>
#include <vector>

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

        Edge(const Edge& e) : pair<T, T>(e)
        {}

        Edge(T first, T second) : pair<T, T>(first, second)
        {}
};

template <Print T>
ostream& operator << (ostream& os, const set<T>& s)
{
    os << "{";
    for (const T& e : s)
        os << e << ", ";
    os << (s.empty() ? "" : "\b\b") << "}";
    return os;
}

template <Print T>
ostream& operator << (ostream& os, const vector<T>& v)
{
    for (const T& e : v)
        os << e << " ";
    if (!v.empty())
        os << "\b";
    return os;
}

template <Print T>
ostream& operator << (ostream& os, const vector<vector<T>>& v)
{
    for (const vector<T>& e : v)
        os << e << "\n";
    if (!v.empty())
        os << "\b";
    return os;
}

set<node> operator - (const set<node>& s1, const set<node>& s2);

#endif
