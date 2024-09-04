# Internet Routing
## Esecuzione sequenziale
### Requisiti
gcc versione 10
### Configurazione
#define SEQ in /src/algo/settings.h
## Esecuzione parallela con OpenMP
### Requisiti
gcc versione 10
OpenMP installato
### Configurazione
#define PAR_OMP in /src/algo/settings.h
## Esecuzione parallela con CUDA
### Requisiti
gcc versione 10
cuda versione ??
### Configurazione
#define PAR_CUDA in /src/algo/settings.h
## Compilazione
lanciare il comando make nella root del progetto
## Esecuzione
lanciare il comando ./main nella root del progetto
  
