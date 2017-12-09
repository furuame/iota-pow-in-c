#ifndef POW_SSE_H_
#define POW_SSE_H_

#include "trinary.h"

typedef struct _pwork_struct Pwork_struct;

struct _pwork_struct {
    char *mid;
    int mwm;
    char *nonce;
    int n;
    long long int ret;
};

Trytes *PowAVX(Trytes *trytes, int mwm);

#endif
