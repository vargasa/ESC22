#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>

std::ostream& operator<<(std::ostream& os, std::vector<int> const& c);
std::vector<int> make_vector(int N);

void printVector(std::vector<int> const& v)
{
    std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
}

int main()
{
  // create a vector of N elements, generated randomly
  int const N = 5;
  std::vector<int> v = make_vector(N);
  std::cout << v << '\n';

  // multiply all the elements of the vector
  // use std::accumulate

  auto product = [](const int a, const int b) { return a*b; }; // using const int& is actually worse ... a pointer is larger than an integer... etc

  auto productTemplate = [](auto a, auto b) {return a*b;}; // kind of a templated version

 std::cout << "Accumulate Product: " << std::accumulate(v.begin(),v.end(),1.,product) << "\n";

  // sort the vector in descending order
  // use std::sort

  auto isGreater = [](const int& a, const int& b) { return a > b; };
  std::sort(v.begin(),v.end(),isGreater);

  std::cout << "Descending sort \n";
  printVector(v);

  // move the even numbers at the beginning of the vector
  // use std::partition

  auto isEven = [](const int& n) { return n%2 == 0; };
  std::partition(v.begin(),v.end(),isEven);
  printVector(v);

  // create another vector with the squares of the numbers in the first vector
  // use std::transform

  auto square = [](const int n) { return n*n; };

  std::vector<int> tv(v.size());
  std::transform(v.begin(),v.end(),tv.begin(),square);

  printVector(tv);
  // find the first multiple of 3 or 7
  // use std::find_if

  auto multipleOf3or7 = [](auto n){ 
    return (n%3 == 0) or (n%7 == 0);
  };

  const auto it = std::find_if(v.begin(),v.end(),multipleOf3or7);

  std::string output = (it == v.end()) ? "Not Found!" : std::to_string(*it);

  std::cout << "First multiple of 3 or 7: " << output << "\n"; 

  // erase from the vector all the multiples of 3 or 7
  // use std::remove_if followed by vector::erase
  std::remove_if(v.begin(),v.end(),multipleOf3or7);
  printVector(v);

}

std::ostream& operator<<(std::ostream& os, std::vector<int> const& c)
{
  os << "{ ";
  std::copy(
            std::begin(c),
            std::end(c),
            std::ostream_iterator<int>{os, " "}
            );
  os << '}';

  return os;
}

std::vector<int> make_vector(int N)
{
  // define a pseudo-random number generator engine and seed it using an actual
  // random device
  std::random_device rd;
  std::default_random_engine eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> dist{1, MAX_N};

  std::vector<int> result;
  result.reserve(N);
  std::generate_n(std::back_inserter(result), N, [&] { return dist(eng); });

  return result;
}
