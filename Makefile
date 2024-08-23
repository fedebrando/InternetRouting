
CC = g++
CFLAGS = -std=c++20
OMP = -fopenmp
ROUTING = routing
UTILITIES = $(ROUTING)/utilities
METRICS = $(ROUTING)/metrics
ALGO = $(ROUTING)/algo
SEMIRING = $(ROUTING)/semiring
OTHER = -I$(UTILITIES) -I$(METRICS) -I$(ALGO) -I$(SEMIRING)

OUT = main
 
all: $(OUT)

main: main.o distance.o bandwidth.o reliability.o utilities.o
	$(CC) $(OMP) $(CFLAGS) -o $@ $^
 
main.o: main.cpp utilities.o $(METRICS)/metrics.hpp $(ALGO)/routing.hpp $(SEMIRING)/lex_product.hpp
	$(CC) $(OTHER) $(OMP) $(CFLAGS) -c -o $@ $<

distance.o: $(METRICS)/distance.cpp $(METRICS)/distance.hpp $(SEMIRING)/semiring.hpp
	$(CC) -I$(SEMIRING) $(CFLAGS) -c -o $@ $<

bandwidth.o: $(METRICS)/bandwidth.cpp $(METRICS)/bandwidth.hpp $(SEMIRING)/semiring.hpp
	$(CC) -I$(SEMIRING) $(CFLAGS) -c -o $@ $<

reliability.o: $(METRICS)/reliability.cpp $(METRICS)/reliability.hpp $(SEMIRING)/semiring.hpp
	$(CC) -I$(SEMIRING) $(CFLAGS) -c -o $@ $<

utilities.o: $(UTILITIES)/utilities.cpp $(UTILITIES)/utilities.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

$(SEMIRING)/lex_product.hpp: utilities.o $(SEMIRING)/semiring.hpp
	
$(METRICS)/metrics.hpp: distance.o bandwidth.o reliability.o

$(ALGO)/routing.hpp: utilities.o $(ALGO)/path.hpp

$(ALGO)/path.hpp: utilities.o $(ALGO)/edge.hpp
	
$(ALGO)/edge.hpp: utilities.o
 
clean:
	rm -f *.o *~ $(OUT)
