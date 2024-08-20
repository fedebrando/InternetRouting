
#include "path.h"

Path::Path() : path() {}

Path::Path(edge e)
{
    path.push_back(e);
}

bool Path::in(const vector<node>& v, node n)
{
    for (node nn : v)
        if (nn == n)
            return true;
    return false;
}

bool Path::loop_free()
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

void Path::insert(edge e)
{
    if (!path.empty())
        if (path.back().second != e.first)
            return;
    path.push_back(e);
}

Path Path::operator + (const Path& other) const
{
    Path p;

    if (!(path.empty() || other.path.empty()))
        if (path.back().second != other.path.front().first)
            throw runtime_error("Cannot concat non contiguous paths!");

    p.path.insert(p.path.end(), path.begin(), path.end());
    p.path.insert(p.path.end(), other.path.begin(), other.path.end());
    return p;
}

bool Path::operator < (const Path& other) const
{
    return path < other.path;
}

bool Path::operator == (const Path& other) const
{
    return path == other.path;
}

ostream& operator << (ostream& os, const Path& p)
{
    for (edge e : p.path)
        os << "(" << e.first << ", " << e.second << ")";
    return os;
}
