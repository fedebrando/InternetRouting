
CC = g++
STD = -std=c++20
OMP = -fopenmp
SRC = src
UTILITIES = $(SRC)/utilities
ALGO = $(SRC)/algo
METRICS = $(SRC)/metrics
SEMIRING = $(SRC)/semiring
R_DATA = $(SRC)/reading_data
OTHER = -I$(ALGO) -I$(METRICS) -I$(SEMIRING) -I$(UTILITIES) -I$(R_DATA)

BUILD = build
OBJ = $(BUILD)/obj
BIN = $(BUILD)/bin

OUT = $(BIN)/main

all: create_dirs $(OUT)

create_dirs:
	mkdir -p $(OBJ) $(BIN)

$(BIN)/main: $(OBJ)/main.o $(OBJ)/distance.o $(OBJ)/bandwidth.o $(OBJ)/reliability.o $(OBJ)/utilities.o $(OBJ)/node.o $(OBJ)/reading_data.o
	$(CC) $(OMP) $(STD) -o $@ $^

$(OBJ)/main.o: $(SRC)/main.cpp $(ALGO)/routing.hpp $(ALGO)/path.hpp $(ALGO)/edge.hpp $(ALGO)/settings.h $(METRICS)/metrics.hpp $(SEMIRING)/semiring.hpp $(SEMIRING)/lex_product.hpp $(UTILITIES)/utilities.hpp $(R_DATA)/reading_data.hpp $(METRICS)/metric_hyperparams.h $(UTILITIES)/node.hpp
	$(CC) $(OMP) $(STD) $(OTHER) -c -o $@ $<

$(OBJ)/reliability.o: $(METRICS)/reliability.cpp $(METRICS)/reliability.hpp $(SEMIRING)/semiring.hpp
	$(CC) $(OMP) $(STD) $(OTHER) -c -o $@ $<

$(OBJ)/bandwidth.o: $(METRICS)/bandwidth.cpp $(METRICS)/bandwidth.hpp $(SEMIRING)/semiring.hpp
	$(CC) $(OMP) $(STD) $(OTHER) -c -o $@ $<

$(OBJ)/distance.o: $(METRICS)/distance.cpp $(METRICS)/distance.hpp $(SEMIRING)/semiring.hpp
	$(CC) $(OMP) $(STD) $(OTHER) -c -o $@ $<

$(OBJ)/utilities.o: $(UTILITIES)/utilities.cpp $(UTILITIES)/utilities.hpp
	$(CC) $(OMP) $(STD) $(OTHER) -c -o $@ $<

$(OBJ)/node.o: $(UTILITIES)/node.cpp $(UTILITIES)/node.hpp
	$(CC) $(OMP) $(STD) $(OTHER) -c -o $@ $<

$(OBJ)/reading_data.o: $(R_DATA)/reading_data.cpp $(R_DATA)/reading_data.hpp $(UTILITIES)/node.hpp $(METRICS)/metrics.hpp $(METRICS)/metric_hyperparams.h $(SEMIRING)/lex_product.hpp $(UTILITIES)/utilities.hpp $(SEMIRING)/semiring.hpp
	$(CC) $(OMP) $(STD) $(OTHER) -c -o $@ $<

clean:
	rm -rf $(BUILD)
