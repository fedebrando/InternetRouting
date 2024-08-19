CC = g++
 
OUT = main
 
all: $(OUT)
 
main: main.o path.o
	$(CC) -o $@ $^ 
 
main.o: main.cpp
	$(CC) -c -o $@ $<
 
path.o: path.cpp path.h
	$(CC) -c -o $@ $<
 
clean:
	rm -f *.o *~ $(OUT)