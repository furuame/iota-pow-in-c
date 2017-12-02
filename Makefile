constants.o: constants.c
	gcc -Wall -g -c -o $@ $<

trinary.o: trinary.c
	gcc -Wall -g -c -o $@ $<

curl.o: curl.c
	gcc -Wall -g -c -o $@ $<

pow_c.o: pow_c.c
	gcc -Wall -g -c -o $@ $<

pow_sse.o: pow_sse.c
	gcc -Wall -g -c -o $@ $<

C_test: trinary.o trinary_test.c constants.o curl.o pow_c.o
	gcc -Wall -DC -g  -o $@ $^ -lpthread

SSE_test: trinary.o trinary_test.c constants.o curl.o pow_sse.o
	gcc -Wall -DSSE -g -o $@ $^ -lpthread

test: C_test SSE_test
	./C_test
	./SSE_test

main: main.c trinary.o
	gcc -Wall -g -o $@ $^
