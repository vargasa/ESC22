#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <queue>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

struct Vertex {
  bool isVisited() const { return visited_; }
  void visit() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    visited_ = true;
  }
  void reset() { visited_ = false; }

  std::vector<int> neighbors_;
  bool visited_ = false;
};

std::vector<Vertex> buildGraph(int nVertices, int maxNumberEdgesPerVertex,
                               std::mt19937 &engine) {
  std::vector<Vertex> directed_graph(nVertices);
  std::uniform_int_distribution<> adjUniformDist(0, nVertices - 1);
  std::uniform_int_distribution<> nNeighborsUniformDist(
      1, maxNumberEdgesPerVertex);
  int vertexId = 0;
  for (auto &v : directed_graph) {
#ifdef DEBUG_GRAPH
    std::cout << "\nvertex " << vertexId << " connected to: " << std::endl;
#endif
    auto nNeighbors = nNeighborsUniformDist(engine);
    v.neighbors_.reserve(nNeighbors);
    std::unordered_set<int> set;
    while (set.size() < nNeighbors) {
      auto toInsert = adjUniformDist(engine);
      if (toInsert != vertexId)
        set.insert(adjUniformDist(engine));
    }
    v.neighbors_.insert(v.neighbors_.end(), set.begin(), set.end());
#ifdef DEBUG_GRAPH
    for (int i = 0; i < nNeighbors; ++i) {
      std::cout << "\t" << v.neighbors_[i];
    }
    std::cout << "\n";
#endif
    vertexId++;
  }
  return directed_graph;
}

void bfs(std::vector<Vertex> &graph, std::vector<int> &distances,
         const int rootIndex) {
  assert(rootIndex < graph.size());
  std::vector<int> queue;
  std::vector<int> nextdistanceQueue;
  queue.push_back(rootIndex);
  distances[rootIndex] = 0;
  graph[rootIndex].visited_ = true;

  int distance = 1;

  while (!queue.empty()) {
#ifdef DEBUG_GRAPH
    std::cout << "distance " << distance << std::endl;
#endif
    for (auto vIndex : queue) {
      auto &myVertex = graph[vIndex];
#ifdef DEBUG_GRAPH
      std::cout << "visiting neighbors of " << vIndex << std::endl;
#endif
      for (auto n : myVertex.neighbors_) {
        if (!graph[n].isVisited() or
            distance < distances[n]) // This condition will help you in parallel
                                     // processing
        {
          // visiting the node and setting its distance
          graph[n].visit();
          distances[n] = distance;
          // pushing back the neighbors so that we can explore its neighbors
          nextdistanceQueue.push_back(n);
#ifdef DEBUG_GRAPH
          std::cout << "neighbor " << n << "\t";
#endif
        }
      }
#ifdef DEBUG_GRAPH
      std::cout << "\n";
#endif
    }
    // all the elements on this distance have been explored. Let's replace the
    // queue with the queue of the next distance.

    distance++;
    queue = nextdistanceQueue;
    nextdistanceQueue.clear();
  }
}

void recursive_visit(std::vector<Vertex> &graph, std::vector<int> &distances,
                     std::queue<int> &queue, const int distance) {
  if (queue.empty()) {
    return;
  }

  // dequeue front node and print it
  int myVertexId = queue.front();
  auto &myVertex = graph[myVertexId];
  queue.pop();

#ifdef DEBUG_GRAPH
  std::cout << "\nvisiting neighbors of " << myVertexId << std::endl;
#endif

  // do for every edge (v, u)
  for (auto n : myVertex.neighbors_) {
    {
      if (!graph[n].isVisited() or distance < distances[n]) {
        // mark it as discovered and enqueue it
#ifdef DEBUG_GRAPH
        std::cout << "visiting " << n << " distance " << distance << "\t";
#endif
        graph[n].visit();
        distances[n] = distance;
        queue.push(n);
      }
    }

    recursive_visit(graph, distances, queue, distance + 1);
  }
}
void recursive_bfs(std::vector<Vertex> &graph, std::vector<int> &distances,
                   const int rootIndex) {
  assert(rootIndex < graph.size());
  std::queue<int> queue;
  queue.push(rootIndex);
  distances[rootIndex] = 0;
  graph[rootIndex].visited_ = true;
  recursive_visit(graph, distances, queue, 1);
};

void parallel_bfs(std::vector<Vertex> &graph, std::vector<int> &distances,
                  const int rootIndex) {}

int main() {
  std::mt19937 engine;
  const int nVertices = 20;
  const int maxNumberEdgesPerVertex = 5;

  // building a direct graph with nVertices, each connected to at most
  // maxNumberEdgesPerVertex
  auto graph = buildGraph(nVertices, maxNumberEdgesPerVertex, engine);

  // We want to compute the distance of each vertex from the vertex at
  // rootIndex. The distance is defined as the minimum amount of vertices to
  // visit in order to reach the vertex from the root.
  int rootIndex = 12;
  // iterative BFS
  std::vector<int> bfs_distances(nVertices, -1);
  auto start = std::chrono::steady_clock::now();
  bfs(graph, bfs_distances, rootIndex);
  auto stop = std::chrono::steady_clock::now();

  auto deltaT = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
  std::cout << "iterative implementation: "<<  deltaT << " milliseconds" << std::endl;

  for (auto &v : graph) {
    v.reset();
  }

  // recursive BFS
  std::vector<int> recursive_bfs_distances(nVertices, -1);

  start = std::chrono::steady_clock::now();
  recursive_bfs(graph, recursive_bfs_distances, rootIndex);
  stop = std::chrono::steady_clock::now();
  deltaT = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
  std::cout << "Recursive implementation: "<<  deltaT << " milliseconds" << std::endl;

  // clearing the graph
  for (auto &v : graph) {
    v.reset();
  }

  // Your parallel BFS goes here
  std::vector<int> parallel_bfs_distances(nVertices, -1);

  start = std::chrono::steady_clock::now();
  parallel_bfs(graph, parallel_bfs_distances, rootIndex);
  stop = std::chrono::steady_clock::now();
  deltaT = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
  std::cout << "Parallel implementation: "<<  deltaT << " milliseconds" << std::endl;


  for (int i = 0; i < nVertices; ++i) {
#ifdef DEBUG_GRAPH
    std::cout << bfs_distances[i] << "\t" << recursive_bfs_distances[i]
              << std::endl;
#endif
    assert(bfs_distances[i] == recursive_bfs_distances[i]);
  }
  std::cout << "Correct!" << std::endl;
}
