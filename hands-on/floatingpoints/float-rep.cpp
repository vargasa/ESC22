#include <iostream>
#include <cstring>
#include <bitset>
#include <string>

int main()
{
  for (float f; std::cin >> f; ) {
    int i;
    std::memcpy(&i, &f, 4);
    auto s = std::bitset<32>(i).to_string();
    std::cout << s[0] << ' ' << s.substr(1, 8) << ' ' << s.substr(9) << '\n';
  }
}
