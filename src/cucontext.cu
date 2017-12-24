#include "cucontext.h"
#include <stdio.h>


void init_cu_devices(CUContext *ctx)
{
    
    cudaSetDevice(ctx->device);
    cudaDeviceProp deviceProp;
    cudaError_t error = cudaGetDeviceProperties(&deviceProp, ctx->device);
    if (error != cudaSuccess) {
        printf("cucintext.cu: GetDeviceProperties failed: %d\n", error);
        exit(0);
    }
    ctx->num_cores = deviceProp.multiProcessorCount;
    ctx->max_memory = deviceProp.totalGlobalMem;
    ctx->num_work_group = deviceProp.maxThreadsPerBlock;

}

void init_cu_buffer(CUContext *ctx)
{
    long mem = 0, max_mem = 0;
    for (int i = 0; i < ctx->kernel_info.num_buffers; i++){
        mem = ctx->kernel_info.buffer_info[i].size;
        if (ctx->kernel_info.buffer_info[i].init_flags & 2) {
            mem *= ctx->num_cores * ctx->num_work_group;
            if (mem > ctx->max_memory) {
                int temp = ctx->max_memory / ctx->kernel_info.buffer_info[i].size;
                ctx->num_cores = temp;
                mem = temp * ctx->kernel_info.buffer_info[i].size;
            }
        }
        // Check Memory bound  
        max_mem += mem;
        if (max_mem >= ctx->max_memory) {
            printf("Max memory pass\n");
            exit(0);
        }
        // cudaMalloc
        cudaError_t error;
        printf("buffer %d need size: %d\n", i, mem);
        switch(i) {
            case 0:
                error = cudaMalloc(&ctx->trit_hash, mem);
                break;
            case 1:
                error = cudaMalloc(&ctx->mid_low, mem);
                break;
            case 2:
                error = cudaMalloc(&ctx->mid_high, mem);
                break;
            case 3:
                error = cudaMalloc(&ctx->state_low, mem);
                break;
            case 4:
                error = cudaMalloc(&ctx->state_high, mem);
                break;
            case 5:
                error = cudaMalloc(&ctx->min_weight_magnitude, mem);
                break;
            case 6:
                error = cudaMalloc(&ctx->found, mem);
                break;
            case 7:
                error = cudaMalloc(&ctx->nonce_probe, mem);
                break;
            case 8:
                error = cudaMalloc(&ctx->loop_count, mem);
                break;
            default:
                continue;
        }
        if (error != cudaSuccess) {
            printf("cucontext.cu: No available memory for Device: %d\n", error);
            exit(0);
        }
    }
}

int init_cucontext(CUContext **ctx)
{
    *ctx = (CUContext *) malloc(sizeof(CUContext));

    if(!(*ctx)) {
        printf("cucontext.cu: init_cucontext: unavailable malloc\n");
        exit(0);
    }
    (*ctx)->device = 0;
    (*ctx)->kernel_info.num_buffers = 9;

    init_cu_devices(*ctx);


    return 1;
}

