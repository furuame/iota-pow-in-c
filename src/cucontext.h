#ifndef CUCONTEXT_H_
#define CUCONTEXT_H_

#define MAX_NUM_BUFFERS 9
#define MAX_NUM_KERNEL 3

typedef long bc_trit_t;

typedef struct {
    size_t size;
    size_t init_flags;
} BufferInfo;

typedef struct {
    BufferInfo buffer_info[MAX_NUM_BUFFERS];
    size_t num_buffers;
} KernelInfo;

typedef struct {
    int device;
    int num_cores;
    //kernel variable
    char* trit_hash;
    bc_trit_t* mid_low;
    bc_trit_t* mid_high;
    bc_trit_t* state_low;
    bc_trit_t* state_high;
    size_t* min_weight_magnitude;
    char* found;
    bc_trit_t* nonce_probe;
    size_t* loop_count;



    size_t max_memory;
    size_t num_work_group;
    KernelInfo kernel_info;
} CUContext;

int init_cucontext(CUContext **ctx);
void init_cu_buffer(CUContext *ctx);
#endif