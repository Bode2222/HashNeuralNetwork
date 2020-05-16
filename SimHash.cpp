#include "SimHash.h"

SimHash::SimHash(int numOfBits, int numOfTables) {
	SetVars(numOfBits, numOfTables);
}

void SimHash::SetVars(int numOfBits, int numOfTables) {
	numTables = numOfTables;
	kbits = numOfBits;
	if (kbits >= pow(2, 8 * sizeof(int))) {
		cout << "kbits over size limit, set to 1 bit per entry" << endl;
		kbits = 1;
	}
	tables = vector<unordered_map<int, vector<int>>>(numTables);
}

void SimHash::CreateTableHashes(int tableIndex, int DimensionOfInput) {//Assign HashTable: pick <kbit> random numbers for each table and sort them
	tableHashes.push_back(vector<unsigned>());
	for (unsigned j = 0; j < kbits; j++) {
		unsigned a = (unsigned)rand();
		unsigned cd = (unsigned)max * DimensionOfInput;
		tableHashes.back().push_back(a % (cd));
	}
	std::sort(tableHashes[tableIndex].begin(), tableHashes[tableIndex].end());
}

void SimHash::Hash(const vector<vector<unsigned>>& points) {
	//set dimension
	d = points[0].size();
	//set max
	for (unsigned i = 0; i < points.size(); i++) {
		for (unsigned j = 0; j < points[i].size(); j++) {
			if (points[i][j] > max) max = points[i][j];
		}
	}

	for (unsigned i = 0; i < numTables; i++) {
		if (i + 1 > tableHashes.size()) {
			CreateTableHashes(i, d);
		}

		//Hash all elements using the random numbers gotten above
		for (unsigned j = 0; j < points.size(); j++) {
			int index = pointToHashIndex(points[j], tableHashes[i]);
			tables[i][index].push_back(j);
		}
	}
}

void SimHash::UpdateTables(const vector<vector<unsigned>>& points) {
	tableHashes.clear();
	for (unsigned i = 0; i < numTables; i++) {
		tables[i].clear();
	}
	Hash(points);
}

int SimHash::pointToHashIndex(const vector<unsigned>& point, const vector<unsigned>& hashVals) {
	int index = 0;
	for (int i = 0; i < kbits; i++) {
		int pointToTest = (hashVals[i]) / max;
		int test = (hashVals[i] % max == 0) ? max : (hashVals[i] % max);
		index += ((point[pointToTest] >= test) ? 1 : 0) * pow(2, (kbits - 1) - i);
	}
	return index;
}

set<int> SimHash::fullQuery(const vector<unsigned>& q) {
	set<int> result;
	if (q.size() != d) {
		cout << "SimHash query function has incorrect dimensions!" << endl;
	}

	for (int i = 0; i < tables.size(); i++) {
		unsigned index = pointToHashIndex(q, tableHashes[i]);
		for (int j = 0; j < tables[i][index].size(); j++) {
			result.insert(tables[i][index][j]);
		}
	}
	return result;
}

set<int> SimHash::randQuery(const vector<unsigned>& q) {
	set<int> result;
	if (q.size() != d) {
		cout << "SimHash query function has incorrect dimensions!" << endl;
	}

	int i = rand() % numTables;
	int index = pointToHashIndex(q, tableHashes[i]);
	for (int j = 0; j < tables[i][index].size(); j++) {
		result.insert(tables[i][index][j]);
	}
	return result;
}

set<int> SimHash::randQueryTill(const vector<unsigned>& q, int target) {
	if (q.size() != d) {
		cout << "SimHash query function has incorrect dimensions!" << endl;
	}
	vector<int> randIndex;
	for (int i = 0; i < numTables; i++) {
		randIndex.push_back(i);
	}
	std::random_shuffle(randIndex.begin(), randIndex.end());

	set<int> result;
	int i = 0;
	int chosen = 0;
	while (i < tables.size() && chosen < target) {
		int index = pointToHashIndex(q, tableHashes[randIndex[i]]);
		std::random_shuffle(tables[randIndex[i]][index].begin(), tables[randIndex[i]][index].end());
		for (int j = 0; j < tables[randIndex[i]][index].size(); j++) {
			if (result.insert(tables[randIndex[i]][index][j]).second) chosen++;
			if (chosen >= target) break;
		}
		i++;
	}
	return result;
}

void SimHash::printTables() {
	for (int i = 0; i < tables.size(); i++) {
		std::cout << "Table " << i << endl;
		//Print hash vals
		cout << "[";
		for (int j = 0; j < tableHashes[i].size(); j++) {
			std::cout << tableHashes[i][j];
			if (j + 1 < tableHashes[i].size()) cout << " ";
		}
		cout << "]";
		std::cout << endl;

		//Print hashed table
		for (auto it : tables[i]) {
			std::cout << "{";
			cout << it.first << ": ";
			for (int l = 0; l < it.second.size(); l++) {
				cout << it.second[l];
				if (l + 1 != it.second.size()) std::cout << ", ";
			}
			std::cout << "}" << endl;
		}

	}
}

vector<vector<unsigned>> SimHash::getTableHashes() {
	return tableHashes;
}

void SimHash::setTableHashes(vector<vector<unsigned>>& tabHashs) {
	tableHashes = tabHashs;
}
