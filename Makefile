CC ?= gcc
NVCC := /usr/local/cuda-9.0/bin/nvcc
CFLAGS_common ?= -Wall -Os -std=gnu99
CFLAGS = -g -I/usr/local/cuda/samples/common/inc/
SRC ?= ./src
OUT ?= ./build
OPENCL_LIB ?= /usr/local/cuda-9.0/lib64/

$(OUT)/constants.o: $(SRC)/constants.c $(SRC)/constants.h
	$(CC) $(CFLAGS_common) -c -o $@ $<

$(OUT)/trinary.o: $(SRC)/trinary.c $(SRC)/trinary.h
	$(CC) $(CFLAGS_common) -c -o $@ $<

$(OUT)/curl.o: $(SRC)/curl.c $(SRC)/curl.h
	$(CC) $(CFLAGS_common) -c -o $@ $<

$(OUT)/pow_c.o: $(SRC)/pow_c.c $(SRC)/pow_c.h
	$(CC) $(CFLAGS_common) -c -o $@ $<

$(OUT)/pow_sse.o: $(SRC)/pow_sse.c $(SRC)/pow_sse.h
	$(CC) $(CFLAGS_common) -msse2 -c -o $@ $<

$(OUT)/pow_avx.o: $(SRC)/pow_avx.c $(SRC)/pow_avx.h
	$(CC) $(CFLAGS_common) -mavx -mavx2 -c -o $@ $<

$(OUT)/pow_cl.o: $(SRC)/pow_cl.c $(SRC)/pow_cl.h
	$(CC) $(CFLAGS_common) -c -o $@ $<

$(OUT)/pow_kernel_cu.o: $(SRC)/pow_kernel_cu.cu $(SRC)/pow_kernel_cu.h $(OUT)
	$(NVCC) $(CFLAGS) -c -o $@ $<

$(OUT)/pow_cu.o: $(SRC)/pow_cu.cu $(SRC)/pow_cu.h $(OUT)
	$(NVCC) $(CFLAGS) -c -o $@ $<

$(OUT)/clcontext.o: $(SRC)/clcontext.c $(SRC)/clcontext.h
	$(CC) $(CFLAGS_common) -c -o $@ $<

$(OUT)/cucontext.o: $(SRC)/cucontext.cu $(SRC)/cucontext.h
	$(NVCC) $(CFLAGS) -c -o $@ $<

pow_c: $(OUT)/trinary.o $(SRC)/trinary_test.c $(OUT)/constants.o $(OUT)/curl.o $(OUT)/pow_c.o
	$(CC) $(CFLAGS_common) -DC -o $@ $^ -lpthread

pow_sse: $(OUT)/trinary.o $(SRC)/trinary_test.c $(OUT)/constants.o $(OUT)/curl.o $(OUT)/pow_sse.o
	$(CC) $(CFLAGS_common) -msse2 -DSSE -g -o $@ $^ -lpthread

pow_avx: $(OUT)/trinary.o $(SRC)/trinary_test.c $(OUT)/constants.o $(OUT)/curl.o $(OUT)/pow_avx.o
	$(CC) $(CFLAGS_common) -mavx -mavx2 -DAVX -g -o $@ $^ -lpthread

pow_cl: $(OUT)/trinary.o $(OUT)/clcontext.o $(SRC)/trinary_test.c $(OUT)/constants.o $(OUT)/curl.o $(OUT)/pow_cl.o
	$(CC) $(CFLAGS_common) -DCL -g -L/usr/local/lib/ -L$(OPENCL_LIB) -o $@ $^ -lOpenCL

pow_cu: $(OUT)/trinary.o $(OUT)/cucontext.o $(SRC)/trinary_test_CUDA.cu $(OUT)/constants.o $(OUT)/curl.o $(OUT)/pow_kernel_cu.o $(OUT)/pow_cu.o
	$(NVCC) $(CFLAGS) -DCU -g -o $@ $^ 

test: pow_c pow_sse pow_avx pow_cl pow_cu
	./pow_c
	./pow_sse
	./pow_avx
	./pow_cl
	./pow_cu

clean:
	rm $(OUT)/*.o pow_c pow_sse pow_avx
