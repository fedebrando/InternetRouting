
#include "utilities.hpp"

set<node> operator - (const set<node>& s1, const set<node>& s2)
{
    set<node> difference;

    set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), inserter(difference, difference.begin()));
    return difference;
}
