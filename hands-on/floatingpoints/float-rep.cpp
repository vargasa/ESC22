#include <iostream>
#include <cstring>
#include <bitset>
#include <string>

int convertBinaryToDecimal(long long n){
    int dec = 0, i = 0, rem;

  while (n!=0) {
    rem = n % 10;
    n /= 10;
    dec += rem * pow(2, i);
    ++i;
  }

  return dec;
}

int main()
{
  for (float f; std::cin >> f; ) {
    int i;
    static_assert(sizeof(int)>=sizeof(float));
    std::memcpy(&i, &f, sizeof(float));
    auto s = std::bitset<32>(i).to_string();
    std::cout <<  "\n\tSign:" << s[0] << "\n\tShifted Exponent: " << s.substr(1, 8) << "\n\tMantissa result (excluding x_0) " << s.substr(9) << '\n';
  }
}
