#include <list>
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

        bool in(const vector<node>& v, node n)
        {
            for (node nn : v)
                if (nn == n)
                    return true;
            return false;
        }

    public:
        Path() : path() {}

        bool loop_free()
        {
            vector<node> v;

            if (path.empty())
                return true;
            v.push_back(path[0].first);
            for (edge e : path)
            {
                if (in(v, e.second))
                    return false;
                v.push_back(e.second);
            }
            return true;
        }

        void insert(edge e)
        {
            if (!path.empty())
                if (path.back().second != e.first)
                    return;
            path.push_back(e);
        }

        Path operator + (const Path& other) const
        {
            Path p;

            if (path.back().second != other.path.front().first)
                throw runtime_error("Cannot concat non contiguous paths!");

            p.path.insert(p.path.end(), path.begin(), path.end());
            p.path.insert(p.path.end(), other.path.begin(), other.path.end());
            return p;
        }
        
        friend ostream& operator << (ostream& os, const Path& p)
        {
            for (edge e : p.path)
                os << "(" << e.first << ", " << e.second << ")";
            return os;
        }

        ~Path() {}
};
