---
title: Use of algorithms and function objects
layout: main
category: cpp
---

The goal of these exercises is to get some confidence with the C++ entities that
are _callable_: functions, function objects and lambdas. You will use them to
customize the behaviour of some of the algorithms provided by the C++ standard
library.

## Exercise 1

The program [`algo.cpp`]({{site.exercises_repo}}/hands-on/cpp/algo.cpp)
generates and prints a vector of randomly-chosen integer numbers.

Build it and run it a few times to verify that it works as
expected:

    [student@esc ~]$ cd esc/hands-on/cpp
    [student@esc cpp]$ g++ -O3 -Wall -Wextra -o algo algo.cpp
    [student@esc cpp]$ ./algo

Then extend the program in order to:

* sum all the elements of the vector
* compute the average of the first half and of the second half of the vector
* move the three central numbers to the beginning
* remove duplicate elements

The source code contains some hints about what algorithms can be used
to implement the requested functionality.
[cppreference.com](http://en.cppreference.com/w/cpp/algorithm) is an excellent source of
documentation for the algorithms (and C++ in general).

## Exercise 2

Consider the program
[`algo_par.cpp`]({{site.exercises_repo}}/hands-on/cpp/algo_par.cpp) and extend
it in order to:

* sum the elements of the vector with the `std::reduce` algorithm, using the
  sequential execution policy
* sum the elements of the vector with the `std::reduce` algorithm, using the
  parallel execution policy
* sort the elements of the vector with the `std::sort` algorithm, using the
  sequential execution policy
* sort the elements of the vector with the `std::sort` algorithm, using the
  parallel execution policy

Compare the execution times.

The use of the parallel algorithms requires:

* compiling with gcc v. 9
* linking against the Threading Building Blocks (TBB) library

To enable the use of gcc v. 9 load the corresponding module:

    [student@esc cpp]$ module load compilers/gcc-9.2.0_sl7
    [student@esc cpp]$ g++ --version

Be sure the output includes `g++ (GCC) 9.2.0`

To enable the use of TBB:

    [student@esc cpp]$ source /shared/software/tbb2019_20191006oss/bin/tbbvars.sh intel64 linux auto_tbbroot

To compile and link:

     [student@esc cpp]$ g++ -O3 algo_par.cpp -std=c++17 -ltbb

TBB will be better introduced in subsequent lectures on parallel programming.

## Exercise 3

Modify the program
[`algo_functions.cpp`]({{site.exercises_repo}}/hands-on/cpp/algo_functions.cpp)
to:

* multiply all the elements of the vector
* sort the vector in descending order
* move the even numbers to the beginning
* create another vector with the squares of the numbers in the first vector
* find the first multiple of 3 or 7
* erase from the vector all the multiples of 3 or 7

For the implementation follow the hints in the source code.
