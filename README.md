# Internet Routing
There are three versions of the Routing Algorithm:
* sequential (`openmp` folder, requires definition of only the `SEQ` macroconstant in `openmp/src/algo/settings.h`)
* parallel with OpenMP (folder `openmp`, requires definition of only the macroconstant `PAR_OMP` in `openmp/src/algo/settings.h`)
* Parallel with CUDA (`cuda` folder).
## Sequential and Parallel with OpenMP
Check if you have at least version `10.5.0` of the `gcc` compiler.
```bash
gcc --version
```
From the `openmp` folder, run the compile command of the project (after setting mutually exclusive macros appropriately).
```bash
make
```
From the `openmp/build/bin` folder, launch the execution of the main program, redirecting the output to a file for better understanding of the results.
```bash
./main > results.txt
```
The choice of metrics for execution is made in the file `openmp/src/metrics/metric_hyperparams.h`.
## Parallel with CUDA
In reference to the University of Parma cluster, the commands for proper execution are given below.\
Import at least version `6.3.0` of the `gcc` compiler.
```bash
module load gcc/6.3.0
```
Import the CUDA module.
```bash
module load cuda
```
Compile the project inside the `cuda` folder (default GPU A100).
```bash
make
```
Run the command to request execution on the cluster (the result will be provided in the text file named `IR-CUDA`).
```bash
sbatch run-main-a100.sh
```
The choice of metrics for execution is made in the file `cuda/src/metrics/metric_hyperparams.h`.
