---
title: Pi approximation
layout: main
section: floatingpoints
---
File: `hands-on/floatingpoints/pi_error.cc`
Trascendental numbers can not be represented as floating-points.
1. Find a way to compute the approximation error of `M_PI`
- You can think of `\pi = M_PI + p`  (`p` is a very small number)
2. Try also with `quadmath`
 - `M_PIq`
 - Most of the functions in quadmath have the same name as `STL` functions with an additional `q` at the end
 - [Quadmath documentation](https://gcc.gnu.org/onlinedocs/gcc-8.1.0/libquadmath.pdf) for documentation 
 - Note: Look how to use `quadmath_snprintf` to print a quad-precision variable
