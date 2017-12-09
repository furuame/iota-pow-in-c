#include <stdio.h>

typedef union {double d; unsigned long long l;} dl;

#define HBITS ( ( (dl) 0xFFFFFFFFFFFFFFFFuLL ).d )
#define LOW40  ( ( (dl)0xFFFFFFFFFFFFFFFFuLL ).d ) //0b1111111111111111111111111111111111111111111111111111111111111111



int main() {
    printf("%f\n", LOW40);
    return 0;
}
