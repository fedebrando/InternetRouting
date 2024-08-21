CC = g++
CFLAGS = -std=c++20
OMP = -fopenmp
METRICS = metrics
ALGO = algorithm
OTHER = -I$(ALGO) -I$(METRICS)

OUT = main
 
all: $(OUT)

main: main.o distance.o
	$(CC) $(OMP) $(CFLAGS) -o $@ $^
 
main.o: main.cpp $(METRICS)/distance.cpp $(ALGO)/path.hpp $(ALGO)/semiring.hpp $(ALGO)/routing.hpp
	$(CC) $(OTHER) $(OMP) $(CFLAGS) -c -o $@ $<

distance.o: $(METRICS)/distance.cpp $(METRICS)/distance.hpp
	$(CC) $(OTHER) $(CFLAGS) -c -o $@ $<
 
clean:
	rm -f *.o *~ $(OUT)