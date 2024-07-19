#include <bits/stdc++.h>
using namespace std;

template <typename T>
void CLOG(T x) {  // CLOG(): clog wrapper
    clog << x;
}
template <typename T>
void CLOG(vector<T> v);
template <typename T1, typename T2>
void CLOG(map<T1, T2> m);
template <typename T1, typename T2>
void CLOG(unordered_map<T1, T2> m);
template <typename T1, typename T2>
void CLOG(pair<T1, T2> p);
void CLOG_MAP(auto m);
template <typename T1, typename T2>
void CLOG(map<T1, T2> m) {
    CLOG_MAP(m);
}
template <typename T1, typename T2>
void CLOG(unordered_map<T1, T2> m) {
    CLOG_MAP(m);
}
template <typename T1, typename T2>
void CLOG(pair<T1, T2> p) {
    clog << '(';
    CLOG(p.first);
    clog << ", ";
    CLOG(p.second);
    clog << ')';
}
template <typename T>
void CLOG(vector<T> v) {
    clog << '[';
    for (int i = 0; i < v.size(); i++) {
        if (i != 0) {
            clog << ", ";
        }
        CLOG(v[i]);
    }
    clog << ']';
}
void CLOG_MAP(auto m) {  // CLOG() for map & unordered_map
    clog << '{';
    for (auto it = m.begin(); it != m.end(); ++it) {
        if (it != m.begin()) {
            clog << ", ";
        }
        CLOG(it->first);
        clog << ": ";
        CLOG(it->second);
    }
    clog << '}';
}
void PRINT(auto x) {
#ifndef ONLINE_JUDGE
    clog.rdbuf(cout.rdbuf());
#endif
    CLOG(x);
    clog << endl;
}
void PRINT(auto x, auto... args) {  // python-like print
#ifndef ONLINE_JUDGE
    clog.rdbuf(cout.rdbuf());
#endif
    CLOG(x);
    clog << ", ";
    PRINT(args...);
}
void PRINT_ARR(auto* arr, size_t n) {  // python-like print for array
#ifndef ONLINE_JUDGE
    clog.rdbuf(cout.rdbuf());
#endif
    clog << '[';
    for (int i = 0; i < n; i++) {
        if (i != 0) {
            clog << ", ";
        }
        CLOG(arr[i]);
    }
    clog << ']' << endl;
}

template <>
struct std::hash<pair<int, int>> {
    size_t operator()(const pair<int, int>& p) const {
        // copy from boost::hash_combine
        size_t x = hash<int>{}(p.first) + 0x9e3779b9 + hash<int>{}(p.second);
        const size_t m = 0xe9846af9b1a615d;
        x ^= x >> 32;
        x *= m;
        x ^= x >> 32;
        x *= m;
        x ^= x >> 28;
        return x;
    }
};

struct TrieNode {
    unordered_map<char, TrieNode*> children;
    bool isWord = false;
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() { root = new TrieNode(); }

    void insert(string word) {
        TrieNode* p = root;

        for (int i = 0; i < word.size(); i++) {
            unordered_map<char, TrieNode*>::iterator iter =
                p->children.find(word[i]);

            if (iter == p->children.end()) {
                p->children[word[i]] = new TrieNode();
            }

            p = p->children[word[i]];
        }

        p->isWord = true;
    }

    bool search(string word) {
        TrieNode* p = root;

        for (int i = 0; i < word.size(); i++) {
            if (p->children.find(word[i]) == p->children.end()) {
                return false;
            }

            p = p->children[word[i]];
        }

        return p->isWord;
    }

    bool startsWith(string prefix) {
        TrieNode* p = root;

        for (int i = 0; i < prefix.size(); i++) {
            if (p->children.find(prefix[i]) == p->children.end()) {
                return false;
            }

            p = p->children[prefix[i]];
        }

        return true;
    }
};

class UnionFind {
private:
    int *root, *rank;

public:
    int num_root;

    UnionFind(int n) {
        root = new int[n];  // root array
        rank = new int[n]();
        num_root = n;

        for (int i = 0; i < n; i++) {
            root[i] = i;
        }
    }

    int find(int x) {
        if (x != root[x]) {  // x is not the root node
            root[x] = find(root[x]);
        }

        return root[x];
    }

    void unionSet(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX != rootY) {
            num_root--;

            if (rank[rootX] > rank[rootY]) {
                root[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                root[rootX] = rootY;
            } else {
                root[rootY] = rootX;
                rank[rootX] += 1;
            }
        }
    }
};

bool topo_DFS(int v, vector<bool>* visited, vector<bool>* curr_visited,
              vector<int>* order, vector<int>* graph) {
    (*visited)[v] = true;
    (*curr_visited)[v] = true;

    for (int i = 0; i < graph[v].size(); i++) {
        int u = graph[v][i];

        if ((*curr_visited)[u]) {
            return false;
        }

        if (!(*visited)[u]) {
            if (!topo_DFS(u, visited, curr_visited, order, graph)) {
                return false;
            }
        }
    }

    (*curr_visited)[v] = false;

    order->push_back(v);

    return true;
}

vector<int> topo(int n, vector<int>* graph) {
    vector<bool> visited(n);
    vector<bool> curr_visited(n);
    vector<int> order;

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            if (!topo_DFS(i, &visited, &curr_visited, &order, graph)) {
                return {};
            }
        }
    }

    return order;
}

// the two children of idx are heaps, idx isn't
template <typename Iter>
void heapify(Iter begin, Iter end, int idx) {
    int len = end - begin;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;

    int max_idx = idx;
    if ((left < len) && *(begin + left) > *(begin + max_idx)) {
        max_idx = left;
    }
    if ((right < len) && *(begin + right) > *(begin + max_idx)) {
        max_idx = right;
    }

    if (max_idx != idx) {
        swap(*(begin + max_idx), *(begin + idx));
        heapify(begin, end, max_idx);
    }
}

template <typename Iter>
void heapsort(Iter begin, Iter end) {
    int len = end - begin;
    for (int i = len / 2 - 1; i >= 0; i--) {
        heapify(begin, end, i);
    }

    while (begin != end) {
        --end;
        swap(*begin, *end);
        heapify(begin, end, 0);
    }
}

template <typename Iter>
void mergesort(Iter begin, Iter end) {
    int len = end - begin;
    if (len <= 1) {  // 1 element
        return;
    }

    Iter mid = begin + len / 2;
    mergesort(begin, mid);  // [begin, mid-1]
    mergesort(mid, end);    // [mid, end)

    // merge
    Iter it1 = begin;
    Iter it2 = mid;
    typename Iter::value_type* copy = new Iter::value_type[len];
    for (int i = 0; i < len; i++) {
        if (it1 != mid && (it2 == end || *it1 < *it2)) {
            copy[i] = *it1;
            ++it1;
        } else {
            copy[i] = *it2;
            ++it2;
        }
    }
    swap_ranges(begin, end, copy);
}

template <typename Iter>
void quicksort(Iter begin, Iter end) {
    int len = end - begin;
    if (len <= 1) {  // 1 element
        return;
    }

    // partition
    Iter pivot = begin;  // the idx which pivot will be
    for (Iter it = begin; it != end - 1; ++it) {
        if (*it < *(end - 1)) {
            swap(*pivot, *it);
            ++pivot;
        }
    }
    swap(*pivot, *(end - 1));

    quicksort(begin, pivot);    // [begin, pivot-1]
    quicksort(pivot + 1, end);  // [mid, end)
}

bool* SieveOfEratosthenes(int n) {  // find prime[2:n-1]
    bool* prime = new bool[n];
    fill_n(prime, n, true);

    for (int i = 2; i < sqrt(n); i++) {
        if (prime[i]) {
            for (int j = i * i; j < n; j += i) {
                prime[j] = false;
            }
        }
    }
    return prime;
}

int main() { return 0; }