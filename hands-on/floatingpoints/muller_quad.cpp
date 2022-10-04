//  This is a perverse example constructed by Jean-Michel Muller.  See section
//  1.3.2 of "Handbook of Floating-Point Arithmetic".
//
//  For an analysis of what happens, see §5 in "How Futile are Mindless Assessments
//  of Roundoff in Floating-Point Computation ?" by W. Kahan at
//  http://www.eecs.berkeley.edu/~wkahan/Mindless.pdf

#include <stdio.h>
#include <iostream>
#include <quadmath.h>
#include <string>
int main( int argc, char* argv[] ) {
    
    char u[256], v[256], w[256], s[256];
    __float128 uq, vq, wq;
    int i, max;
    printf( "n = " );
    scanf( "%d", &max);
    printf( "u0 = " );
    scanf( "%s", &u );
    printf( "u1 = " );
    scanf( "%s", &v );
    uq = strtoflt128(u, NULL);

    vq = strtoflt128(v, NULL);

    quadmath_snprintf(s, 256, "%Qf", vq);
    printf( "Computation from 3 to n:\n" );
    for ( i = 3; i <= max; i++ ) {
	wq = 111.0 - 1130.0 / vq + 3000.0 / ( vq * uq );
	uq = vq;
	vq = wq;
	quadmath_snprintf(s, 256, "%Qf", vq);
	printf("%s\n", s);
    }
    return 0;
}
