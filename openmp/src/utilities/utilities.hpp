
/*
 * Utilities used by Path and Routing
*/

#ifndef UTILITIES
#define UTILITIES

#include <iostream>
#include <sstream>
#include <set>
#include <concepts>
#include <vector>
#include <algorithm>

using namespace std;

typedef size_t node;

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

template <Print T>
ostream& operator << (ostream& os, const set<T>& s)
{
    ostringstream oss;
    string str;

    oss << "{";
    for (const T& e : s)
        oss << e << ", ";
    str = oss.str();
    if (!s.empty())
    {
        str.pop_back();
        str.pop_back();
    }
    os << str << "}";
    return os;
}

template <Print T>
ostream& operator << (ostream& os, const vector<T>& v)
{
    ostringstream oss;
    string str;

    for (const T& e : v)
        oss << e << " ";
    str = oss.str();
    if (!v.empty())
        str.pop_back();
    os << str;
    return os;
}

template <Print T>
ostream& operator << (ostream& os, const vector<vector<T>>& v)
{
    ostringstream oss;
    string str;

    for (const vector<T>& e : v)
        oss << e << "\n";
    str = oss.str();
    if (!v.empty())
        str.pop_back();
    os << str;
    return os;
}

set<node> operator - (const set<node>& s1, const set<node>& s2);

#endif
