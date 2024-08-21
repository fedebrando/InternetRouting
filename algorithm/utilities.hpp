
#ifndef UTILITIES
#define UTILITIES

#include <iostream>
#include <set>
#include <concepts>

using namespace std;

typedef unsigned int node;

set<node> operator - (const set<node>& s1, const set<node>& s2);

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

#endif