constants.o: constants.c
	gcc -Wall -g -c -o $@ $<

trinary.o: trinary.c
	gcc -Os -Wall -g -c -o $@ $<

curl.o: curl.c
	gcc -Os -Wall -g -c -o $@ $<

pow_c.o: pow_c.c
	gcc -Os -Wall -g -c -o $@ $<

pow_sse.o: pow_sse.c
	gcc -Os -Wall -g -msse2 -c -o $@ $<

pow_avx.o: pow_avx.c
	gcc -Os -Wall -g -mavx -mavx2 -c -o $@ $<

C_test: trinary.o trinary_test.c constants.o curl.o pow_c.o
	gcc -Wall -Os -DC -g  -o $@ $^ -lpthread

SSE_test: trinary.o trinary_test.c constants.o curl.o pow_sse.o
	gcc -Wall -Os -msse2 -DSSE -g -o $@ $^ -lpthread

AVX_test: trinary.o trinary_test.c constants.o curl.o pow_avx.o
	gcc -Wall -Os -mavx -mavx2 -DAVX -g -o $@ $^ -lpthread

test: C_test SSE_test AVX_test
	./C_test
	./SSE_test
	./AVX_test

clean:
	rm *.o C_test SSE_test AVX_test
