
#include <iostream>
#include <vector>
#include "routing.hpp"
#include "metrics.hpp"
#include "lex_product.hpp"
#include "utilities.hpp"
#include "node.hpp"
#include "reading_data.hpp"

using namespace std;

int main(void) 
{
    vector<Node> v = getV("../../data/node.dat");
#ifdef SHORTEST_WIDEST
    vector<vector<LexProduct<Distance, Bandwidth>>> a(v.size(), vector<LexProduct<Distance, Bandwidth>>(v.size()));
#else
    vector<vector<LexProduct<Distance, Reliability>>> a(v.size(), vector<LexProduct<Distance, Reliability>>(v.size()));
#endif

    getA("../../data/edge.dat", v, a);

#ifdef SHORTEST_WIDEST
    Routing<LexProduct<Distance, Bandwidth>, Node> r(v, a);
#else
    Routing<LexProduct<Distance, Reliability>, Node> r(v, a);
#endif

    r.compute();
    cout << r.getD() << endl;
    cout << r.getPi() << endl;

    return 0; 
}
