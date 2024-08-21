CC = g++
CFLAGS = -std=c++20
OUT = main
 
all: $(OUT)
 
main: main.o distance.o
	$(CC) -fopenmp $(CFLAGS) -o $@ $^ 
 
main.o: main.cpp distance.cpp
	$(CC) -fopenmp $(CFLAGS) -c -o $@ $<

distance.o: distance.cpp distance.h
	$(CC) $(CFLAGS) -c -o $@ $<
 
clean:
	rm -f *.o *~ $(OUT)