#include "pow_cu.h"
#include "pow_kernel_cu.h"
#include "cucontext.h"
extern "C"{
#include "curl.h"
#include "trinary.h"
}
#include "constants.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#define HASH_LENGTH 243              //trits
#define NONCE_LENGTH 81              //trits
#define STATE_LENGTH 3 * HASH_LENGTH //trits
#define TRANSACTION_LENGTH 2673 * 3 
#define HIGH_BITS 0xFFFFFFFFFFFFFFFF
#define LOW_BITS 0x0000000000000000
#define LOW_0 0xDB6DB6DB6DB6DB6D
#define HIGH_0 0xB6DB6DB6DB6DB6DB
#define LOW_1 0xF1F8FC7E3F1F8FC7
#define HIGH_1 0x8FC7E3F1F8FC7E3F
#define LOW_2 0x7FFFE00FFFFC01FF
#define HIGH_2 0xFFC01FFFF803FFFF
#define LOW_3 0xFFC0000007FFFFFF
#define HIGH_3 0x003FFFFFFFFFFFFF

void init_BufferInfo(CUContext *ctx)
{
    ctx->kernel_info.buffer_info[0] = (BufferInfo){sizeof(char) * HASH_LENGTH};
    ctx->kernel_info.buffer_info[1] = (BufferInfo){sizeof(int64_t) * STATE_LENGTH, 2};
    ctx->kernel_info.buffer_info[2] = (BufferInfo){sizeof(int64_t) * STATE_LENGTH, 2};
    ctx->kernel_info.buffer_info[3] = (BufferInfo){sizeof(int64_t) * STATE_LENGTH, 2};
    ctx->kernel_info.buffer_info[4] = (BufferInfo){sizeof(int64_t) * STATE_LENGTH, 2};
    ctx->kernel_info.buffer_info[5] = (BufferInfo){sizeof(size_t)};
    ctx->kernel_info.buffer_info[6] = (BufferInfo){sizeof(char)};
    ctx->kernel_info.buffer_info[7] = (BufferInfo){sizeof(int64_t), 2};
    ctx->kernel_info.buffer_info[8] = (BufferInfo){sizeof(size_t)};

    init_cu_buffer(ctx);
}

void write_cu_buffer(CUContext *ctx, int64_t *mid_low, int64_t *mid_high, int mwm, int loop_count)
{

    BufferInfo *buffer_info = ctx->kernel_info.buffer_info;
    cudaError_t error;

    error = cudaMemcpy(ctx->mid_low, mid_low, buffer_info[1].size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("pow_cu.cu: toGpu failed: %d\n",error);
        exit(0);
    }
    error = cudaMemcpy(ctx->mid_high, mid_high, buffer_info[2].size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("pow_cu.cu: toGpu failed: %d\n",error);
        exit(0);
    }
    error = cudaMemcpy(ctx->min_weight_magnitude, &mwm, buffer_info[5].size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("pow_cu.cu: toGpu failed: %d\n",error);
        exit(0);
    }
    error = cudaMemcpy(ctx->loop_count, &loop_count, buffer_info[8].size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("pow_cu.cu: toGpu failed: %d\n",error);
        exit(0);
    }
}

void init_state(char *state, int64_t *mid_low, int64_t *mid_high, size_t offset)
{
    for (int i = 0; i < STATE_LENGTH; i++) {
        switch (state[i]) {
            case 0:
                    mid_low[i] = HIGH_BITS;
                    mid_high[i] = HIGH_BITS;
                    break;
            case 1:
                    mid_low[i] = LOW_BITS;
                    mid_high[i] = HIGH_BITS;
                    break;
            default:
                    mid_low[i] = HIGH_BITS;
                    mid_high[i] = LOW_BITS;
        }
    }
    mid_low[offset] = LOW_0;
    mid_high[offset] = HIGH_0;
    mid_low[offset + 1] = LOW_1;
    mid_high[offset + 1] =  HIGH_1;
    mid_low[offset + 2] = LOW_2;
    mid_high[offset + 2] = HIGH_2;
    mid_low[offset + 3] = LOW_3;
    mid_high[offset + 3] = HIGH_3;
}

char *pwork(char *state, int mwm){

    char found = 0;
    size_t block_dim, grid_dim;
    CUContext *titan = NULL;

    //init device
    init_cucontext(&titan);
    printf("test - num_cores: %d, max_mem: %d\n", titan->num_cores, titan->max_memory);

    init_BufferInfo(titan);
    

    //init kernel dim
    grid_dim = titan->num_cores;
    block_dim = STATE_LENGTH;
    while (block_dim > titan->num_work_group) {
        block_dim /= 3;
    }

    //init states
    int64_t mid_low[STATE_LENGTH] = {0}, mid_high[STATE_LENGTH] = {0};
    init_state(state, mid_low, mid_high, HASH_LENGTH - NONCE_LENGTH);

    //copy data to device
    write_cu_buffer(titan, mid_low, mid_high, mwm, 32);

    //launch init kernel
    Launch_init_kernel(titan, block_dim, grid_dim);

    cudaDeviceSynchronize();
    //launch search kernel loop
    while (found == 0) {
        Launch_search_kernel(titan, block_dim, grid_dim);

        //cudaDeviceSynchronize();
        cudaError_t error = cudaMemcpy(&found, titan->found, sizeof(char), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            printf("pow_cu.cu: toCpu failed: %d\n",error);
            exit(0);
        }
    }

    //launch finalize kernel
    Launch_finalize_kernel(titan, block_dim, grid_dim);
    

    char *buf = (char*)malloc(HASH_LENGTH * sizeof(char));

    if (found > 0) {
        cudaError_t error = cudaMemcpy(buf, titan->trit_hash, HASH_LENGTH * sizeof(char), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            printf("pow_cu.cu: toCpu failed: %d\n",error);
            exit(0);
        }
    }
    return buf;
}


Trytes *PowCU(Trytes *trytes, int mwm)
{
	Curl *c = NewCurl();

    /* Preapre trytes passing to curl */
    char tyt[(transactionTrinarySize - HashSize) / 3] = {0};
    for (int i = 0; i < (transactionTrinarySize - HashSize) / 3; i++) {
        tyt[i] = trytes->data[i];
    }
    Trytes *inn = NULL;
    init_Trytes(&inn);
    inn->toTrytes(inn, tyt, (transactionTrinarySize - HashSize) / 3);

    c->Absorb(c, inn);

    Trits *tr = trytes->toTrits(trytes);

    char *c_state = (char *) malloc(c->state->len);
    /* Prepare an array storing tr[transactionTrinarySize - HashSize:] */
    for (int i = 0; i < tr->len - (transactionTrinarySize - HashSize); i++) {
        int idx = transactionTrinarySize - HashSize + i;
        c_state[i] = tr->data[idx];
        
    }
    for (int i =  tr->len - (transactionTrinarySize - HashSize); i < c->state->len; i++) {
        c_state[i] = c->state->data[i];
    }
    
    char *ret = pwork(c_state, mwm);


    memcpy(&tr->data[TRANSACTION_LENGTH - HASH_LENGTH], ret, HASH_LENGTH * sizeof(char));
    return tr->toTrytes(tr);
}
