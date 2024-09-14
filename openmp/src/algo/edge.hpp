
#ifndef EDGE
#define EDGE

#include <utility>
#include "utilities.hpp"

using namespace std;

/*
 * Class Edge<T> IS-A std::pair<T, T>
*/
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

#endif
