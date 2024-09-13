
#include <iostream>
#include <vector>
#include "routing.hpp"
#include "metrics.hpp"
#include "lex_product.hpp"
#include "utilities.hpp"

/* include del nuovo main*/
#include <fstream>
#include <iostream> 
#include <string> 
#include <sstream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include "node.hpp"

#define SHORTEST_WIDEST 

using namespace std;

vector<string> split(const string& str, char delimiter) 
{
    vector<string> tokens;
    string token;
    stringstream ss(str);

    while (getline(ss, token, delimiter)) 
        tokens.push_back(token);
    return tokens;
}

int main(void) 
{
    ifstream inputFile;  
    vector<string> values;
    string line; 
    vector<Node> v;
#ifdef SHORTEST_WIDEST
    vector<vector<LexProduct<Distance, Bandwidth>>> a;
    LexProduct<Distance, Bandwidth> lp;
#else
    vector<vector<LexProduct<Distance, Reliability>>> a;
    LexProduct<Distance, Reliability> lp;
#endif
    
    int first;
    int second;

    inputFile.open("../../data/node.dat");
    if (!inputFile) 
    { 
        cerr << "Error opening \"data.dat\" file!" << endl;  
        return 1; 
    } 
    getline(inputFile, line); // to skip header line
    while (getline(inputFile, line)) 
    {
        values = split(line, ',');
        v.push_back(Node(values[2], values[5], values[3], stod(values[1]), stod(values[4])));
    }
    inputFile.close(); 
#ifdef SHORTEST_WIDEST
    a = vector<vector<LexProduct<Distance, Bandwidth>>>(v.size(), vector<LexProduct<Distance, Bandwidth>>(v.size()));
#else
    a = vector<vector<LexProduct<Distance, Reliability>>>(v.size(), vector<LexProduct<Distance, Reliability>>(v.size()));
#endif

    inputFile.open("../../data/edge.dat");
    if (!inputFile) 
    { 
        cerr << "Error opening \"edge.dat\" file!" << endl; 
        return 1; 
    } 
    getline(inputFile, line); // to skip header line
    while (getline(inputFile, line)) 
    {
        values = split(line, ',');
        first = stoi(values[0]);
        second = stoi(values[1]);
#ifdef SHORTEST_WIDEST
        lp = LexProduct<Distance, Bandwidth>(Distance(haversine(v[first], v[second])), Bandwidth(stod(values[3])));
        a[first][second] = lp;
        a[second][first] = lp;
#else
        lp = LexProduct<Distance, Reliability>(Distance(haversine(v[first], v[second])), Reliability(stod(values[2])));
        a[first][second] = lp;
        a[second][first] = lp;
#endif
    }
    inputFile.close(); 

#ifdef SHORTEST_WIDEST
    Routing<LexProduct<Distance, Bandwidth>, Node> r(v, a);
#else
    Routing<LexProduct<Distance, Reliability>, Node> r(v, a);
#endif

    r.compute();
    cout << r.getD() << endl;
    cout << r.getPi() << endl;

    return 0; 


    //vector<string> v = {"0", "1", "2", "3", "4"};
    //vector<vector<METRIC>> a(v.size(), vector<METRIC>(v.size()));
    // distance del paper
    /*
    {
        {Distance::zero(), Distance(1), Distance::zero(), Distance::zero(), Distance::zero()},
        {Distance(1), Distance::zero(), Distance(2), Distance(1), Distance(1)},
        {Distance::zero(), Distance(2), Distance::zero(), Distance(1), Distance(1)},
        {Distance::zero(), Distance(1), Distance(1), Distance::zero(), Distance::zero()},
        {Distance::zero(), Distance(1), Distance(1), Distance::zero(), Distance::zero()}
    };
    
    
    // distance video yt
    {
        {Distance::zero(), Distance(1), Distance::zero(), Distance::zero(), Distance::zero(), Distance(3)},
        {Distance(1), Distance::zero(), Distance(3), Distance::zero(), Distance(5), Distance(1)},
        {Distance::zero(), Distance(3), Distance::zero(), Distance(2), Distance::zero(), Distance::zero()},
        {Distance::zero(), Distance::zero(), Distance(2), Distance::zero(), Distance(1), Distance(6)},
        {Distance::zero(), Distance(5), Distance::zero(), Distance(1), Distance::zero(), Distance(2)},
        {Distance(3), Distance(1), Distance::zero(), Distance(6), Distance(2), Distance::zero()}
    };
    */
    

    // distance * reliability
    /*
    a[0][1] = METRIC(Distance(1), Reliability(0.5));
    a[1][4] = METRIC(Distance(1), Reliability(0.1));
    a[1][3] = METRIC(Distance(1), Reliability(0.9));
    a[3][2] = METRIC(Distance(1), Reliability(0.9));
    a[4][2] = METRIC(Distance(1), Reliability(1));
    a[1][2] = METRIC(Distance(2), Reliability(0.81));
    a[1][0] = METRIC(Distance(1), Reliability(0.5));
    a[4][1] = METRIC(Distance(1), Reliability(0.1));
    a[3][1] = METRIC(Distance(1), Reliability(0.9));
    a[2][3] = METRIC(Distance(1), Reliability(0.9));
    a[2][4] = METRIC(Distance(1), Reliability(1));
    a[2][1] = METRIC(Distance(2), Reliability(0.81));
    */

    // distance * bandwidth
    /*
    a[0][1] = METRIC(Distance(1), Bandwidth(10));
    a[1][0] = METRIC(Distance(1), Bandwidth(10));
    a[1][2] = METRIC(Distance(2), Bandwidth(90));
    a[2][1] = METRIC(Distance(2), Bandwidth(90));
    a[1][3] = METRIC(Distance(1), Bandwidth(5));
    a[3][1] = METRIC(Distance(1), Bandwidth(5));
    a[1][4] = METRIC(Distance(1), Bandwidth(100));
    a[4][1] = METRIC(Distance(1), Bandwidth(100));
    a[3][2] = METRIC(Distance(1), Bandwidth(100));
    a[2][3] = METRIC(Distance(1), Bandwidth(100));
    a[4][2] = METRIC(Distance(1), Bandwidth(100));
    a[2][4] = METRIC(Distance(1), Bandwidth(100));
    */

    /*
    a[0][1] = METRIC(Distance(1));
    a[1][0] = METRIC(Distance(1));
    a[1][2] = METRIC(Distance(2));
    a[2][1] = METRIC(Distance(2));
    a[1][3] = METRIC(Distance(1));
    a[3][1] = METRIC(Distance(1));
    a[1][4] = METRIC(Distance(1));
    a[4][1] = METRIC(Distance(1));
    a[3][2] = METRIC(Distance(1));
    a[2][3] = METRIC(Distance(1));
    a[4][2] = METRIC(Distance(1));
    a[2][4] = METRIC(Distance(1));
    */


    // bandwidth * distance
    /*
    a[0][1] = METRIC(Bandwidth(5), Distance(1));
    a[2][1] = METRIC(Bandwidth(5), Distance(4));
    a[3][0] = METRIC(Bandwidth(5), Distance(1));
    a[4][0] = METRIC(Bandwidth(10), Distance(5));
    a[4][2] = METRIC(Bandwidth(5), Distance(1));
    a[2][3] = METRIC(Bandwidth(5), Distance(1));
    a[3][4] = METRIC(Bandwidth(10), Distance(1));
    */
   
    /*
    cout << "(" << Distance::zero() << ")" << endl;
    cout << "[" << Bandwidth::zero() << "]" << endl;
    cout << LexProduct<Distance, Bandwidth>(Distance::zero(), Bandwidth::zero()) << endl;
    cout << METRIC::zero() << endl;
    cout << LexProduct<Distance, Bandwidth>::unity() << endl;
    */

    /*
    Routing<METRIC, string> r(v, a);

    r.compute();
    cout << r.getD() << endl;
    cout << r.getPi() << endl;
    */

    //return 0;
}
