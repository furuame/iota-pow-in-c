#include "trinary.h"
#include "constants.h"
#include <stdio.h>

int main()
{
    for (int i = 0; i < 27; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%d ", tryteToTritsMappings[i][j]);
        }
        printf("\n");
    }

    for (int i = 0; i < 27; i++) {
        printf("%c ", TryteAlphabet[i]);
    }
    printf("\n");
    
    Trits *test = NULL;
    init_Trits(&test);
    char c[] = {1, 0, -1};

    test->toTrits(test, c, 3);
    printf("%lld\n", test->Int(test));

    return 0;
}
