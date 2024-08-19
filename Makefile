CC = g++
CFLAGS = -std=c++20
OUT = main
 
all: $(OUT)
 
main: main.o path.o
	$(CC) $(CFLAGS) -o $@ $^ 
 
main.o: main.cpp
	$(CC) $(CFLAGS) -c -o $@ $<
 
path.o: path.cpp path.h
	$(CC) $(CFLAGS) -c -o $@ $<
 
clean:
	rm -f *.o *~ $(OUT)