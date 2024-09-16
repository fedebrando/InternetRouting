CC = gcc
SRC = src
MATH = -lm
SRC = src
UTILITIES = $(SRC)/utilities
R_DATA = $(SRC)/reading_data

BUILD = build
OBJ = $(BUILD)/obj
BIN = $(BUILD)/bin

OUT = $(BIN)/main

all: create_dirs $(OUT)

create_dirs:
	mkdir -p $(OBJ) $(BIN)

$(BIN)/main: $(OBJ)/main.o $(OBJ)/node.o $(OBJ)/reading_data.o
	$(CC) -o $@ $^ $(MATH)

$(OBJ)/main.o: $(SRC)/main.c $(OBJ)/node.o $(OBJ)/reading_data.o
	$(CC) -I$(UTILITIES) -I$(R_DATA) -c -o $@ $<

$(OBJ)/reading_data.o: $(R_DATA)/reading_data.c $(R_DATA)/reading_data.h $(OBJ)/node.o
	$(CC) -I$(UTILITIES) -c -o $@ $<

$(OBJ)/node.o: $(UTILITIES)/node.c $(UTILITIES)/node.h
	$(CC) -c -o $@ $< $(MATH)

clean:
	rm -rf $(BUILD)
