// adapted from http://www.bnikolic.co.uk/blog/hpc-perf-branchprediction.html
// (this version avoids compiler optimization)
// try with -O2 -Ofast and then
// g++ -Ofast -march=native branchPredictor.cpp -funroll-all-loops
#include <algorithm>
#include <iostream>
#include <vector>

int main()
{
  bool sorted = true;
  // generate data
  const size_t arraySize = 32768;
  std::vector<int> test(arraySize);
  std::vector<int> data(arraySize);

  for (unsigned c = 0; c < arraySize; ++c) {
    test[c] = std::rand() % 256;
    data[c] = std::rand() % 256;
  }

  if (sorted)
    std::sort(test.begin(), test.end());


  for (unsigned i = 0; i < 1000000; ++i) {
    long long sum = 0;

    for (unsigned c = 0; c < arraySize; ++c) {
      if (test[c] >= 128)
        sum += data[c];
    }
  }

  std::cout << "sum = " << sum << ' ' << "sorted: " << (sorted? "yes" : "no") << std::endl;

  return 0;
}
