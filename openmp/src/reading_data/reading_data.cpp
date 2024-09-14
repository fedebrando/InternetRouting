
#include "reading_data.hpp"

vector<string> split(const string& str, char delimiter) 
{
    vector<string> tokens;
    string token;
    stringstream ss(str);

    while (getline(ss, token, delimiter)) 
        tokens.push_back(token);
    return tokens;
}

vector<Node> getV(string filename)
{
    ifstream inputFile;  
    vector<string> values;
    string line; 
    vector<Node> v;
    int first;
    int second;

    inputFile.open(filename);
    if (!inputFile) 
        throw runtime_error("Error in opening file!");
    getline(inputFile, line); // to skip header line
    while (getline(inputFile, line)) 
    {
        values = split(line, ',');
        v.push_back(Node(values[2], values[5], values[3], stod(values[1]), stod(values[4])));
    }
    inputFile.close(); 
    return v;
}

#ifdef SHORTEST_WIDEST
void getA(string filename, const vector<Node>& v, vector<vector<LexProduct<Distance, Bandwidth>>>& a)
{
    ifstream inputFile;  
    vector<string> values;
    string line; 
    int first;
    int second;
    LexProduct<Distance, Bandwidth> lp;

    inputFile.open(filename);
    if (!inputFile) 
        throw runtime_error("Error in opening file!");
    getline(inputFile, line); // to skip header line
    while (getline(inputFile, line)) 
    {
        values = split(line, ',');
        first = stoi(values[0]);
        second = stoi(values[1]);
        lp = LexProduct<Distance, Bandwidth>(Distance(haversine(v[first], v[second])), Bandwidth(stod(values[3])));
        a[first][second] = lp;
        a[second][first] = lp;
    }
    inputFile.close();
}
#else
void getA(string filename, const vector<Node>& v, vector<vector<LexProduct<Distance, Reliability>>>& a)
{
    ifstream inputFile;  
    vector<string> values;
    string line; 
    int first;
    int second;
    LexProduct<Distance, Reliability> lp;

    inputFile.open(filename);
    if (!inputFile) 
        throw runtime_error("Error in opening file!");
    getline(inputFile, line); // to skip header line
    while (getline(inputFile, line)) 
    {
        values = split(line, ',');
        first = stoi(values[0]);
        second = stoi(values[1]);
        lp = LexProduct<Distance, Reliability>(Distance(haversine(v[first], v[second])), Reliability(stod(values[2])));
        a[first][second] = lp;
        a[second][first] = lp;
    }
    inputFile.close(); 
}
#endif
