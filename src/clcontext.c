#include "clcontext.h"
#include <stdio.h>

void init_cl_devices(CLContext *ctx)
{
    cl_uint num_platform = 0;
    cl_int errno;
    cl_platform_id platform;

    errno = clGetPlatformIDs(0, NULL, &num_platform);

    if (errno != CL_SUCCESS) {
        printf("Cannot get the number of OpenCL platforms available.\n");
        exit(0);
    }

    // We only need one Platform
    clGetPlatformIDs(1, &platform, NUL);

    // Get Device IDs
    cl_uint platform_num_device;
    if (clGetDeviceIDs(platform, CLDEVICE_TYPE_GPU, 1, &ctx->device, &platform_num_device) != CL_SUCCESS) {
        printf("Failed to get OpenCL Device IDs for platform.\n");
        exit(0);
    }

    // Create OpenCL context
    ctx->context =  (cl_context) clCreateContext(NULL, 1, &ctx->device, NULL, NULL, &errno);
    if (errno != CL_SUCCESS) {
        printf("Failed to create OpenCL Context\n");
        exit(0);
    }

    // Get Device Info (num_cores)
    if (CL_SUCCESS != clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &ctx->num_cores, NULL)) {
        printf("Failed to get Device info (num_cores)!\n");
        exit(0);
    }

    // Get Device Info (max_memory)
    if (CL_SUCCESS != clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &ctx->max_memory, NULL)) {
        printf("Failed to get Device info (max memory)!"\n);
        exit(0);
    }

    // Get Device Info (num work group)
    if (CL_SUCCESS != clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &ctx->num_work_group, NULL)) {
        ctx->num_work_group = 1;
    }

    // Create Command Queue
    ctx->cmdq = clCreateCommandQueue(ctx->context, ctx->device, 0, &errno);
    if (errno != CL_SUCCESS) {
        printf("Failed to create command queue\n");
        exit(0);
    }

    return 1;
}

void init_cl_program(CLContext *ctx)
{
    FILE *fin = fopen(KERNEL_PATH, "r");
    char *source_str;
    size_t source_size;
    cl_int errno;
    
    if (!fin) {
        printf("OpenCL Kernel File doesn't exist\n");
        exit(0);
    }

    source_str = (char *) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fin);
    fclose(fin);

    ctx->program = clCreateProgramWithSource(ctx->context, ctx->kernel_info.num_src, (const char **) &source_str, (const size_t *) &source_size, &errno);
    if (CL_SUCCESS != errno) {
        printf("Failed to create program with source\n");
        exit(0);
    }

    errno = clBuildProgram(ctx->program, 1, &ctx->device, "-Werror", NULL, NULL);
    if (CL_SUCCESS != errno) {
        printf("Failed to build program\n");
        exit(0);
    }

    return 1;
}

int init_clcontext(CLContext **ctx)
{
    *ctx = (CLContext *) malloc(sizeof(CLContext));

    if(!(*ctx)) {
        printf("clcontext.c: init_clcontext: unavailable malloc\n");
        exit(0);
    }

    (*ctx)->kernel_info.num_buffers = 9;
    (*ctx)->kernel_info.num_kernels = 3;
    (*ctx)->kernel_info.num_src = 1;
    
    init_cl_devices(*ctx);
    init_cl_program(*ctx);

    return 1;
}
