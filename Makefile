CC = g++
CFLAGS = -std=c++20
OMP = -fopenmp
METRICS = metrics
ALGO = algorithm
OTHER = -I$(ALGO) -I$(METRICS)

OUT = main
 
all: $(OUT)

main: main.o distance.o bandwidth.o reliability.o utilities.o
	$(CC) $(OMP) $(CFLAGS) -o $@ $^
 
main.o: main.cpp distance.o bandwidth.o reliability.o utilities.o $(ALGO)/path.hpp $(ALGO)/routing.hpp $(ALGO)/semiring.hpp
	$(CC) $(OTHER) $(OMP) $(CFLAGS) -c -o $@ $<

distance.o: $(METRICS)/distance.cpp $(METRICS)/distance.hpp utilities.o $(ALGO)/semiring.hpp
	$(CC) $(OTHER) $(CFLAGS) -c -o $@ $<

bandwidth.o: $(METRICS)/bandwidth.cpp $(METRICS)/bandwidth.hpp utilities.o $(ALGO)/semiring.hpp
	$(CC) $(OTHER) $(CFLAGS) -c -o $@ $<

reliability.o: $(METRICS)/reliability.cpp $(METRICS)/reliability.hpp utilities.o $(ALGO)/semiring.hpp
	$(CC) $(OTHER) $(CFLAGS) -c -o $@ $<

utilities.o: $(ALGO)/utilities.cpp $(ALGO)/utilities.hpp
	$(CC) $(CFLAGS) -c -o $@ $<
 
clean:
	rm -f *.o *~ $(OUT)