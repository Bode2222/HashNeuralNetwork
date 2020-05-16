#pragma once
#include <time.h>		//rand
#include <vector>		//hold data
#include <set>
#include <cstdlib>		//rand
#include <algorithm>	//sort
#include <unordered_map>//hold data
#include <iostream>
#include <thread>
using namespace std;

/*SimHash Class: A class used for the implementation of a version of locality sensitive hashing. Inspired by "Similarity Search in High Dimensions via Hashing" by gionis et al
*Takes in high dimension points in space and stores their indexes in hash tables, their indexes are either specified one at a time or are assumed when given a vector of points
*Rules:
*All input must be positive whole numbers, shift and multiply where necessary.
*K < 32, because int only holds 4 bytes
*/

class SimHash
{
	void CreateTableHashes(int tableIndex, int DimensionOfInput);
	vector<vector<unsigned>> tableHashes;
	//tables-> mapping the hash value to the index of the input
	vector<unordered_map<int, vector<int>>> tables;
	unsigned kbits = 0, max = 0, numTables = 0, d = 0;
	int pointToHashIndex(const vector<unsigned>& point, const vector<unsigned>& hash);
public:
	void printTables();
	SimHash() {};
	vector<vector<unsigned>> getTableHashes();
	void setTableHashes(vector<vector<unsigned>>& tabHashs);
	//Places point addresses into the hash table
	void Hash(const vector<vector<unsigned>>& points);
	//Places point addresses into the hash table by going through each index in each table and reassignin its values
	void UpdateTables(const vector<vector<unsigned>>& points);
	//Retrieves point indexes in the same bucket as the query point from every table
	set<int> fullQuery(const vector<unsigned>& q);
	//Retrieves point index in the same bucket as the query point from a random table
	set<int> randQuery(const vector<unsigned>& q);
	//Retrieves point indexes in random tables until either we run out of tables or we meet our target number of points
	set<int> randQueryTill(const vector<unsigned>& q, int target);
	SimHash(int k, int numOfTables);
	void SetVars(int k, int numOfTables);
};

