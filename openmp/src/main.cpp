
#include <iostream>
#include <vector>
#include "routing.hpp"
#include "metrics.hpp"
#include "lex_product.hpp"
#include "utilities.hpp"
#include "node.hpp"
#include "reading_data.hpp"
#include "metric_hyperparams.h"

#define TIMING

#ifdef TIMING
#include <chrono>
#endif

using namespace std;

int main(void) 
{
#ifdef TIMING
    chrono::high_resolution_clock::time_point start, end;
    chrono::milliseconds durata_ms;
#endif
    vector<Node> v = getV("../../data/node.dat");
#ifdef WSP
    vector<vector<LexProduct<Distance, Bandwidth>>> a(v.size(), vector<LexProduct<Distance, Bandwidth>>(v.size()));
#else
    vector<vector<LexProduct<Distance, Reliability>>> a(v.size(), vector<LexProduct<Distance, Reliability>>(v.size()));
#endif

    getA("../../data/edge.dat", v, a);

#ifdef WSP
    Routing<LexProduct<Distance, Bandwidth>, Node> r(v, a);
#else
    Routing<LexProduct<Distance, Reliability>, Node> r(v, a);
#endif

#ifdef TIMING
    start = chrono::high_resolution_clock::now();
#endif
    r.compute();
#ifdef TIMING
    end = chrono::high_resolution_clock::now();

    durata_ms = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Elapsed time: " << durata_ms.count() << " ms" << endl << endl;
#endif
    
    /*
    cout << "--- D MATRIX ---" << endl;
    cout << r.getD() << endl << endl;
    cout << "--- PI MATRIX ---" << endl;
    cout << r.getPi() << endl;

    */

    print_results(cout, r.getD(), r.getPi(), v);

    return 0; 
}
