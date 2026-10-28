# PaSTiLa (Parallel Automatic Snippet-based Time series Labeling Algorithm)

This repository is related to the PaSTiLa (Parallel Automatic Snippet-based Time series Labeling Algorithm) algorithm that automates pattern discovery in large time series with a graphics processor. PaSTiLa is authored by Andrey Goglachev (goglachevai@susu.ru) and Mikhail Zymbler (mzym@susu.ru), South Ural State University, Chelyabinsk, Russia. The repository contains the PaSTiLa source code (in Python).

PaSTiLa serves as a wrapper for the **PDSS** (C22dist-based pattern discovery) and **PSF** (MPdist-based pattern discovery) algorithms. Their binaries are not included and must be built separately. PaSTiLa predicts execution time for each segment length, distributes tasks across GPUs for load balancing, executes the binaries, and then selects the optimal segment length by comparing normalized snippet profiles.

## Usage

The commands below use `py` (Windows) and are run from the PaSTiLa root directory. On Linux/macOS, use `python3`.

Install dependencies:

```
py -m pip install -r requirements.txt
```

Build PDSS and PSF from their source code and put the binaries into `src/algorithms/`:

```
src/algorithms/PDSS/PDSS.exe    (PDSS on Linux)
src/algorithms/PSF/PSF.exe      (PSF on Linux)
```

Alternatively, pass the path to the binary with `--exe` (for both `run` and `calibrate`).

Calibrate the execution time model (perform once per machine; not required for single-GPU execution):

```
py src/cli.py calibrate --algo pdss --gpu 0
py src/cli.py calibrate --algo psf  --gpu 0
```

| Parameter             | Description                                                                  |
|-----------------------|------------------------------------------------------------------------------|
| `--algo`              | `pdss` or `psf` (required)                                                   |
| `--gpu`               | Physical GPU id to run the calibration on (default: 0)                       |
| `--exe`               | Path to the PDSS/PSF binary (default: `src/algorithms/<ALGO>/<ALGO>.exe`)    |
| `--out`               | Model file to write (default: `src/models/<algo>_runtime_model.joblib`)      |
| `--csv`               | Also write the raw calibration measurements to this CSV file (optional)      |

Run:

```
py src/cli.py run \
--algo pdss \
--input path/to/series.txt \
--outdir results \
--ts-length 50000 \
--segment-lengths 50:400:50 \
--num-snippets 3 \
--candidate-step 250
```

| Parameter             | Description                                                                  |
|-----------------------|------------------------------------------------------------------------------|
| `--algo`              | `pdss` or `psf` (required)                                                   |
| `--input`             | Text file containing the time series (required)                              |
| `--outdir`            | Directory for results (required)                                             |
| `--ts-length`         | Length of the input time series (required)                                   |
| `--segment-lengths`   | Segment lengths to try: a range `MIN:MAX[:STEP]` (e.g. `50:400:50`) or a comma-separated list (e.g. `50,100,200`) (required) |
| `--num-snippets`      | Number of snippets K to find (required)                                      |
| `--candidate-step`    | Step size between candidate segments, PDSS only (default: the segment length, i.e. no overlap) |
| `--reserve`           | λ: search λ·K patterns per length; after the best length is chosen, the two closest patterns are merged repeatedly until K remain (default: 1, no merging) |
| `--gpus`              | Comma-separated physical GPU ids, e.g. `0,1,2` (default: all GPUs found by `nvidia-smi -L`) |
| `--exe`               | Path to the PDSS/PSF binary (default: `src/algorithms/<ALGO>/<ALGO>.exe`)    |
| `--model`             | Runtime model file (default: `src/models/<algo>_runtime_model.joblib`)       |
| `--mpi`               | Run as master-worker over MPI, see below                                     |

### Cluster run (MPI)

`--mpi` runs PaSTiLa as master-worker over MPI (requires `mpi4py` and an MPI runtime, e.g. `py -m pip install mpi4py impi-rt` on Windows, or MS-MPI / Open MPI).

```
mpiexec -n 4 py src/cli.py run --mpi --algo psf --input path/to/series.txt --outdir results \
    --ts-length 50000 --segment-lengths 50:400:50 --num-snippets 3 --reserve 2
```

## Output

Results are saved in `<outdir>/<algo>/<series>/seglen_<L>/`. Data for the best length is copied to `<outdir>/<algo>/<series>/best_seglen_<L>/`, and the file `<outdir>/<algo>/<series>/summary.json` contains information on the predicted and actual execution times, as well as an estimate of the area between the profiles for each length. `best_seglen_<L>/merge.json` lists which of the λ·K patterns were merged into each final pattern.

## Citation

```
@article{ZymblerGoglachev2024,
 author    = {Mikhail Zymbler and Andrey Goglachev},
 title     = {PaSTiLa: Scalable Parallel Algorithm for Unsupervised Labeling of Long Time Series},
 journal   = {Lobachevskii Journal of Mathematics},
 volume    = {45},
 number    = {3},
 pages     = {1333-1347},
 year      = {2024},
 doi       = {10.1134/S1995080224600766},
 url       = {https://doi.org/10.1134/S1995080224600766}
}
```