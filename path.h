#ifndef PATH
#define PATH

#include <vector>
#include <utility>
#include <iostream>

using namespace std;

typedef unsigned int node;
typedef pair<node, node> edge;

class Path
{
    private:
        vector<edge> path;
        bool in(const vector<node>& v, node n);

    public:
        Path() : path() {}

        bool loop_free();
        void insert(edge e);
        Path operator + (const Path& other) const;
        friend ostream& operator << (ostream& os, const Path& p)
        {
            for (edge e : p.path)
                os << "(" << e.first << ", " << e.second << ")";
            return os;
        }
        ~Path() {}
};

#endif