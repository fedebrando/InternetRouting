
#ifndef READING_DATA
#define READING_DATA

#include <vector>
#include <fstream>
#include <sstream>
#include "node.hpp"
#include "metrics.hpp"
#include "lex_product.hpp"
#include "settings.h"
#include "metric_hyperparams.h"

using namespace std;

vector<string> split(const string& str, char delimiter);
vector<Node> getV(string filename);
#ifdef WSP
void getA(string filename, const vector<Node>& v, vector<vector<LexProduct<Distance, Bandwidth>>>& a);
#else
void getA(string filename, const vector<Node>& v, vector<vector<LexProduct<Distance, Reliability>>>& a);
#endif

#endif
