---
title: Quadratic Equation
layout: main
section: floatingpoint
---
File: `hands-on/floatingpoints/quad_eq.cc`
Let's take a general quadratic equation:
ax^2 + bx + c = 0

1. Implement a program to find the root of the equations with
   - `a = 5*10^(-4)`
   - `b = 100`
   - `c = 5*10^(-3)`
2. Compute the roots manually and compare the results
3. What is going on?

Note that there are two sources of cancellation.
In particular, if $b^2 >> 4ac$ at the numerator you might have catastrophic cancellation! 

4. With some algebra one can rewrite the problematic root to avoid one cancellation
5. Try to rationalize the problematic root and implement the new solution 
6. Compare the results!
