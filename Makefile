CC = g++
STD = -std=c++20
OMP = -fopenmp
SRC = src
UTILITIES = $(SRC)/utilities
ALGO = $(SRC)/algo
METRICS = $(SRC)/metrics
SEMIRING = $(SRC)/semiring

BUILD = build
OBJ = $(BUILD)/obj
BIN = $(BUILD)/bin

OUT = $(BIN)/main

all: create_dirs $(OUT)

create_dirs:
	mkdir -p $(OBJ) $(BIN)

$(BIN)/main: $(OBJ)/main.o $(OBJ)/distance.o $(OBJ)/bandwidth.o $(OBJ)/reliability.o $(OBJ)/utilities.o
	$(CC) $(OMP) $(STD) -o $@ $^

$(OBJ)/main.o: $(SRC)/main.cpp $(ALGO)/routing.hpp $(ALGO)/path.hpp $(ALGO)/edge.hpp $(ALGO)/settings.h $(METRICS)/metrics.hpp $(SEMIRING)/semiring.hpp $(SEMIRING)/lex_product.hpp $(UTILITIES)/utilities.hpp 
	$(CC) $(OMP) $(STD) -I$(ALGO) -I$(METRICS) -I$(SEMIRING) -I$(UTILITIES) -c -o $@ $<

$(OBJ)/reliability.o: $(METRICS)/reliability.cpp $(METRICS)/reliability.hpp $(SEMIRING)/semiring.hpp
	$(CC) $(OMP) $(STD) -I$(SEMIRING) -c -o $@ $<

$(OBJ)/bandwidth.o: $(METRICS)/bandwidth.cpp $(METRICS)/bandwidth.hpp $(SEMIRING)/semiring.hpp
	$(CC) $(OMP) $(STD) -I$(SEMIRING) -c -o $@ $<

$(OBJ)/distance.o: $(METRICS)/distance.cpp $(METRICS)/distance.hpp $(SEMIRING)/semiring.hpp
	$(CC) $(OMP) $(STD) -I$(SEMIRING) -c -o $@ $<

$(OBJ)/utilities.o: $(UTILITIES)/utilities.cpp $(UTILITIES)/utilities.hpp
	$(CC) $(OMP) $(STD) -c -o $@ $<

clean:
	rm -rf $(BUILD)

