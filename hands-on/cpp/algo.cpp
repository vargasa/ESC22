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
  int const N = 4;
  std::vector<int> v = make_vector(N);
  std::cout << v << '\n';

  // sum all the elements of the vector
  // use std::accumulate
  std::cout << "Accumulate " << std::accumulate(v.begin(),v.end(),0) << "\n";

  // compute the average of the first half and of the second half of the vector

  auto itMid = v.begin()+N/2;

  std::cout << "Average of the first part " << std::accumulate(v.begin(),itMid,0)/static_cast<double>(std::distance(v.begin(),itMid)) << "\n";
  std::cout << "Average of the second part " << std::accumulate(itMid,v.end(),0)/static_cast<double>(std::distance(itMid,v.end())) << "\n";

  // move the three central elements to the beginning of the vector
  // use std::rotate

  std::cout << "Rotating Elements:\n";

  std::rotate(v.begin(),v.begin()+3,v.end());

  printVector(v);
  
  // remove duplicate elements
  // use std::sort followed by std::unique/unique_copy

  std::cout << "Working with unique:\n";

  std::sort(v.begin(),v.end());
  std::unique(v.begin(),v.end());

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
  std::random_device rd;
  std::default_random_engine eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> dist{1, MAX_N};

  std::vector<int> result;
  result.reserve(N);
  std::generate_n(std::back_inserter(result), N, [&] { return dist(eng); });

  return result;
}
