//     define bfs_sequential(graph(V,E), source s):
// 2        for all v in V do
// 3            d[v] = -1;
// 4        d[s] = 0; level = 1; FS = {}; NS = {};
// 5        push(s, FS);
// 6        while FS !empty do
// 7            for u in FS do 
// 8                for each neighbour v of u do 
// 9                    if d[v] = -1 then
// 10                       push(v, NS);
// 11                       d[v] = level;
// 12           FS = NS, NS = {}, level = level + 1;


#include <vector>
#include <random>
#include <iostream>
#include <cassert>

struct Vertex 
{

    std::vector<int> neighbors;
    int level = -1;
    bool visited = false;
};

int main()
{
    std::mt19937 engine;
    const int nVertices = 500;
    const int maxNumberEdgesPerVertex = 10;
    int rootIndex = 27;
    assert(rootIndex < nVertices);
    std::vector<Vertex> graph(nVertices);
    std::uniform_int_distribution<> adjUniformDist(0,nVertices-1);
    std::uniform_int_distribution<> nNeighborsUniformDist(1,maxNumberEdgesPerVertex);
    int vertexId = 0;
    for(auto& v: graph)
    {
        std::cout << "\nvertex " << vertexId << " connected to: " << std::endl; 
        auto nNeighbors = nNeighborsUniformDist(engine);
        v.neighbors.reserve(nNeighbors);
        
        for(int i =0; i<nNeighbors; ++i)
        {
            v.neighbors.push_back(adjUniformDist(engine));
            std::cout << "\t" << v.neighbors.back();
        } 
        std::cout << "\n";
        vertexId++;
    }

    auto sequential_bfs = [&graph,=vertexId](){
        
    };

}



