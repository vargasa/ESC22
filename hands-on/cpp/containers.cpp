#include <vector>
#include <list>
#include <chrono>
#include <iostream>
#include <random>
#include <cassert>
#include <cstdlib>

using Duration = std::chrono::duration<float>;

std::default_random_engine eng{std::random_device{}()};
using Distribution = std::uniform_int_distribution<>;
Distribution dist;

/*
Duration fill(std::list<int>& cont, int N){

  auto start = std::chrono::high_resolution_clock::now();
  cont.clear();
  for (size_t i = 0; i != N; ++i){
    auto n = dist(eng,Distribution::param_type{0, static_cast<int>( cont.size())});
    auto it = std::next(cont.begin(),n);
  }

  return std::chrono::high_resolution_clock::now() - start;

}

Duration process(std::list<int> const& cont)
{
  auto start = std::chrono::high_resolution_clock::now();

  // the volatile is to avoid complete removal by the optimizer
  auto volatile v = std::accumulate(std::begin(cont), std::end(cont), 0, [](int a, int n) {
      return a ^ n;
    });
  (void)v; // to silence a warning about unused variable

  return std::chrono::high_resolution_clock::now() - start;
}
*/
template <typename Container>  
//requires std::is_same_v<std::list<int>,Container> or std::is_same_v<std::vector<int>,Container>  //concepts 
Duration fill(Container& cont, int N)
{
  assert(N >= 0);

  auto start = std::chrono::high_resolution_clock::now();

  cont.clear();
  for (size_t i = 0; i != N; ++i) {
    // generate a number between 0 and the current size of the container
    auto n = dist(eng, Distribution::param_type{0, static_cast<int>(cont.size())});
    // advance n positions in the container
    auto it = std::next(cont.begin(), n);
    // insert the number itself in that position
    cont.insert(it, n);
  }
  assert(static_cast<int>(cont.size()) == N);

  return std::chrono::high_resolution_clock::now() - start;
}

template <typename Container>  
Duration process(Container const& cont)
{
  auto start = std::chrono::high_resolution_clock::now();

  // the volatile is to avoid complete removal by the optimizer
  // and v is just to make sure accumulate is executed even if
  // we dont use it anywhere
  auto volatile v = std::accumulate(std::begin(cont), std::end(cont), 0, [](int a, int n) {
      return a ^ n;
    });
  (void)v; // to silence a warning about unused variable

  return std::chrono::high_resolution_clock::now() - start;
}

int main(int argc, char* argv[])
{
  int const N = (argc > 1) ? std::atoi(argv[1]) : 10000;

  std::vector<int> v;
  std::cout << "vector fill: " << fill(v, N).count() << " s\n";
  std::cout << "vector process: " << process(v).count() << " s\n";
  std::list<int> l;
  std::cout << "list fill: " << fill(l, N).count() << " s\n";
  std::cout << "list process: " << process(l).count() << " s\n";
}
