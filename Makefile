CC = g++
STD = -std=c++20
OMP = -fopenmp
UTILITIES = src/utilities
ALGO = src/algo
METRICS = src/METRICS
SEMIRING = src/semiring
SRC = src

OUT = main

all: $(OUT)

main: main.o distance.o bandwidth.o reliability.o utilities.o
	$(CC) $(OMP) $(STD) -o $@ $^

main.o: $(SRC)/main.cpp $(ALGO)/routing.hpp $(ALGO)/path.hpp $(ALGO)/edge.hpp $(ALGO)/settings.h $(METRICS)/metrics.hpp $(SEMIRING)/semiring.hpp $(SEMIRING)/lex_product.hpp $(UTILITIES)/utilities.hpp 
	$(CC) $(OMP) $(STD) -I$(ALGO) -I$(METRICS) -I$(SEMIRING) -I$(UTILITIES) -c -o $@ $<

reliability.o: $(METRICS)/reliability.cpp $(METRICS)/reliability.hpp $(SEMIRING)/semiring.hpp
	$(CC) $(OMP) $(STD) -I$(SEMIRING) -c -o $@ $<

bandwidth.o: $(METRICS)/bandwidth.cpp $(METRICS)/bandwidth.hpp $(SEMIRING)/semiring.hpp
	$(CC) $(OMP) $(STD) -I$(SEMIRING) -c -o $@ $<

distance.o: $(METRICS)/distance.cpp $(METRICS)/distance.hpp $(SEMIRING)/semiring.hpp
	$(CC) $(OMP) $(STD) -I$(SEMIRING) -c -o $@ $<

utilities.o: $(UTILITIES)/utilities.cpp $(UTILITIES)/utilities.hpp
	$(CC) $(OMP) $(STD) -c -o $@ $<

clean:
	rm -f *.o *~ $(OUT)

