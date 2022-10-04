/*
 Code from Sebastiano Fabio Schifano and Enrico Calore
*/

#include <stdio.h>
#include <math.h>
#include <values.h>

/* extract nth LSB from object stored in lvalue x */
#define GET_BIT(x, n) ((((const char *) &x)[LITTLE_ENDIAN ? (n) / CHARBITS : sizeof(x) - (n) / CHARBITS - 1] >> ((n) % CHARBITS)) & 0x01)

#define PUT_BIT(x, n) (putchar(GET_BIT((x), (n)) ? '1' : '0'))

void print_float_bits(float f) {

    int i;

    i = FLOATBITS - 1;
    PUT_BIT(f, i);
    putchar(' ');
    for(i--; i >= 23; i--) {
        PUT_BIT(f, i);
    }
    putchar(' ');
    for(; i >= 0; i--) {
        PUT_BIT(f, i);
    }

}

int main(int argc, char **argv) {

    float f;
    while(scanf("%f", &f) == 1) {
        printf("%10g = %24.17g = ", f, f);
        print_float_bits(f);
        putchar('\n');
    }

}
