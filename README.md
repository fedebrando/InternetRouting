# Internet Routing
Sono presenti tre versioni dell'Algoritmo di Routing:
* sequenziale (cartella `openmp`, richiede la definizione della sola macrocostante `SEQ` in `openmp/src/algo/settings.h`)
* parallela in OpenMP (cartella `openmp`, richiede la definizione della sola macrocostante `PAR_OMP` in `openmp/src/algo/settings.h`)
* parallela in CUDA (cartella `cuda`).
## Sequenziale e Parallela con OpenMP
Verificare se si possiede almeno la versione 10.5.0 del compilatore gcc.
```bash
gcc --version
```
Dalla cartella `openmp` lanciare il comando per la compilazione del progetto (dopo aver impostato opportunamente le macrocostanti mutuamente esclusive).
```bash
make
```
Dalla cartella `openmp/build/bin` lanciare l'esecuzione del programma principale, redirezionando l'output su un file per una maggiore comprensione dei risultati.
```bash
./main > results.txt
```
La scelta della metrica per l'esecuzione avviene nel file `openmp/src/metrics/metric_hyperparams.h`.
## Parallela con CUDA
In riferimento al cluster dell'Università di Parma, i comandi per una corretta esecuzione sono qui sotto riportati.\
Importare almeno la versione 6.3.0 del compilatore gcc.
```bash
module load gcc/6.3.0
```
Importare il modulo CUDA.
```bash
module load cuda
```
Compilare il progetto all'interno della cartella `cuda` (default GPU A100).
```bash
make
```
Lanciare il comando per richiedere l'esecuzione sul cluster (il risultato verrà fornito nel file di testo con nome `IR-CUDA`).
```bash
sbatch run-main-a100.sh
```
La scelta della metrica per l'esecuzione avviene nel file `cuda/src/metrics/metric_hyperparams.h`.
