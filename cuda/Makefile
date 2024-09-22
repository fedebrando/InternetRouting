
CC = nvcc
ARCH = -arch=sm_75#a100
#ARCH = -arch=compute_60#p100
RDC = -rdc=true
CUDA = -lcudadevrt
SRC = src
ALGO = $(SRC)/algo
UTILITIES = $(SRC)/utilities
METRICS = $(SRC)/metrics
R_DATA = $(SRC)/reading_data
OTHER = -I$(ALGO) -I$(UTILITIES) -I$(METRICS) -I$(R_DATA)

BUILD = build
OBJ = $(BUILD)/obj
BIN = $(BUILD)/bin

OUT = $(BIN)/main

all: create_dirs $(OUT)

create_dirs:
	mkdir -p $(OBJ) $(BIN)

$(BIN)/main: $(OBJ)/main.o $(OBJ)/nset.o $(OBJ)/path.o $(OBJ)/pset.o $(OBJ)/lex_product.o $(OBJ)/distance.o $(OBJ)/bandwidth.o $(OBJ)/reliability.o $(OBJ)/wsp_or_mrsp.o $(OBJ)/routing.o $(OBJ)/utilities.o $(OBJ)/reading_data.o $(OBJ)/node.o
	$(CC) $(ARCH) $(RDC) $^ -o $@ $(CUDA)

$(OBJ)/main.o: $(SRC)/main.cu $(UTILITIES)/nset.cuh $(UTILITIES)/path.cuh $(UTILITIES)/pset.cuh $(METRICS)/lex_product.cuh $(METRICS)/wsp_or_mrsp.cuh $(UTILITIES)/utilities.cuh $(R_DATA)/reading_data.cuh $(R_DATA)/node.cuh $(METRICS)/metric_hyperparams.h
	$(CC) $(ARCH) $(OTHER) -dc $< -o $@

$(OBJ)/routing.o: $(ALGO)/routing.cu $(ALGO)/routing.cuh $(UTILITIES)/utilities.cuh $(UTILITIES)/nset.cuh $(UTILITIES)/path.cuh $(UTILITIES)/pset.cuh $(METRICS)/lex_product.cuh $(METRICS)/wsp_or_mrsp.cuh $(METRICS)/metric_hyperparams.h
	$(CC) $(ARCH) $(OTHER) -dc $< -o $@

$(OBJ)/wsp_or_mrsp.o: $(METRICS)/wsp_or_mrsp.cu $(METRICS)/wsp_or_mrsp.cuh $(METRICS)/lex_product.cuh $(METRICS)/distance.cuh $(METRICS)/bandwidth.cuh $(METRICS)/reliability.cuh $(METRICS)/metric_hyperparams.h $(UTILITIES)/utilities.cuh
	$(CC) $(ARCH) $(OTHER) -dc $< -o $@

$(OBJ)/reliability.o: $(METRICS)/reliability.cu $(METRICS)/reliability.cuh $(UTILITIES)/utilities.cuh
	$(CC) $(ARCH) $(OTHER) -dc $< -o $@

$(OBJ)/bandwidth.o: $(METRICS)/bandwidth.cu $(METRICS)/bandwidth.cuh $(UTILITIES)/utilities.cuh
	$(CC) $(ARCH) $(OTHER) -dc $< -o $@

$(OBJ)/distance.o: $(METRICS)/distance.cu $(METRICS)/distance.cuh $(UTILITIES)/utilities.cuh
	$(CC) $(ARCH) $(OTHER) -dc $< -o $@

$(OBJ)/lex_product.o: $(METRICS)/lex_product.cu $(METRICS)/lex_product.cuh $(UTILITIES)/utilities.cuh
	$(CC) $(ARCH) $(OTHER) -dc $< -o $@

$(OBJ)/pset.o: $(UTILITIES)/pset.cu $(UTILITIES)/pset.cuh $(UTILITIES)/utilities.cuh $(UTILITIES)/path.cuh
	$(CC) $(ARCH) $(OTHER) -dc $< -o $@

$(OBJ)/path.o: $(UTILITIES)/path.cu $(UTILITIES)/path.cuh $(UTILITIES)/utilities.cuh
	$(CC) $(ARCH) $(OTHER) -dc $< -o $@

$(OBJ)/nset.o: $(UTILITIES)/nset.cu $(UTILITIES)/nset.cuh $(UTILITIES)/utilities.cuh
	$(CC) $(ARCH) $(OTHER) -dc $< -o $@

$(OBJ)/utilities.o: $(UTILITIES)/utilities.cu $(UTILITIES)/utilities.cuh
	$(CC) $(ARCH) $(OTHER) -dc $< -o $@

$(OBJ)/reading_data.o: $(R_DATA)/reading_data.cu $(R_DATA)/reading_data.cuh $(METRICS)/lex_product.cuh $(METRICS)/metric_hyperparams.h $(METRICS)/wsp_or_mrsp.cuh $(UTILITIES)/utilities.cuh $(R_DATA)/node.cuh 
	$(CC) $(ARCH) $(OTHER) -dc $< -o $@

$(OBJ)/node.o: $(R_DATA)/node.cu $(R_DATA)/node.cuh $(METRICS)/wsp_or_mrsp.cuh $(METRICS)/metric_hyperparams.h $(METRICS)/lex_product.cuh $(UTILITIES)/path.cuh $(UTILITIES)/pset.cuh
	$(CC) $(ARCH) $(OTHER) -dc $< -o $@

clean:
	rm -rf $(BUILD)
