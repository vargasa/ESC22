#include "mpi.h"
#include <iostream>

int main(int argc, char* argv[])
{
  int numranks, rank, dest, source, rc, count, tag = 1;
  int msg_size = 10000;
  int *inmsg = new int[msg_size];
  for (int i = 0; i< msg_size; i++) {
     inmsg[i] = i;
     std::cout << inmsg[i] << std::endl;
  }
  delete[] inmsg;
  return 0;
}

    
//      int inmsg[msg_size];
  //        int outmsg[msg_size];
  //
