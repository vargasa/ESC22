//  -*-  mode: C++;  fill-column: 132  -*-
//  Time-stamp: "Modified on 30-March-2016 at 21:54:42 by jearnold on olhswep03.cern.ch"
//
//
//  This is a perverse example constructed by Jean-Michel Muller.  See section
//  1.3.2 of "Handbook of Floating-Point Arithmetic".
//
//  For an analysis of what happens, see §5 in "How Futile are Mindless Assessments
//  of Roundoff in Floating-Point Computation ?" by W. Kahan at
//  http://www.eecs.berkeley.edu/~wkahan/Mindless.pdf

#include <stdio.h>

int main( int argc, char* argv[] ) {
    double u, v, w;
    int i, max;
    printf( "n = " );
    scanf( "%d", &max );
    printf( "u0 = " );
    scanf( "%lf", &u );
    printf( "u1 = " );
    scanf( "%lf", &v );
    printf( "Computation from 3 to n:\n" );
    for ( i = 3; i <= max; i++ ) {
	w = 111.0 - 1130.0 / v + 3000.0 / ( v * u );
	u = v;
	v = w;
	printf( "u%d=%1.17g\n", i, v );
    }
    return 0;
}
