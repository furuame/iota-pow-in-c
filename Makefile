constants.o: constants.c
	gcc -Wall -g -c -o $@ $<

trinary.o: trinary.c
	gcc -Wall -g -c -o $@ $<

curl.o: curl.c
	gcc -Wall -g -c -o $@ $<

pow_c.o: pow_c.c
	gcc -Wall -g -c -o $@ $<

trinary_test: trinary.o trinary_test.c constants.o curl.o pow_c.o
	gcc -Wall -g  -o $@ $^ -lpthread


main: main.c trinary.o
	gcc -Wall -g -o $@ $^
