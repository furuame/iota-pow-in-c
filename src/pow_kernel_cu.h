#include <stdio.h>
#include <stdlib.h>
#include "cucontext.h"

void Launch_init_kernel(CUContext *ctx, size_t block_dim, size_t grid_dim);
void Launch_search_kernel(CUContext *ctx, size_t block_dim, size_t grid_dim);
void Launch_finalize_kernel(CUContext *ctx, size_t block_dim, size_t grid_dim);