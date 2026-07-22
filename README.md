# PaSTiLa (Parallel Automatic Snippet-based Time series Labeling Algorithm)

This repository is related to the PaSTiLa (Parallel Automatic Snippet-based Time series Labeling Algorithm) algorithm that automates pattern discovery in large time series with a graphics processor. PaSTiLa is authored by Andrey Goglachev (goglachevai@susu.ru) and Mikhail Zymbler (mzym@susu.ru), South Ural State University, Chelyabinsk, Russia. The repository contains the PaSTiLa's source code (in C, CUDA, Python).

PaSTiLa serves as a wrapper for the **PDSS** (C22dist-based pattern discovery) and **PSF** (MPdist-based pattern discovery) algorithms located in the `algorithms/` directory. PaSTiLa predicts execution time for each segment length, distributes tasks across GPUs for load balancing, executes the binaries, and then selects the optimal segment length by comparing normalized snippet profiles.

## Usage

The commands below use `py` (Windows). On Linux/macOS, use `python3`.

Install dependencies:

```
py -m pip install -r requirements.txt
```

Calibrate the execution time model (perform once per machine; not required for single-GPU execution):

```
py cli.py calibrate --algo pdss --gpu 0
py cli.py calibrate --algo psf  --gpu 0
```

Run:

```
py cli.py run \
--algo pdss \
--input path/to/series.txt \
--outdir results \
--ts-length 50000 \
--segment-lengths 50:400:50 \
--num-snippets 3 \
--candidate-step 250
```

- `--segment-lengths` accepts a range in `MIN:MAX[:STEP]` format or a comma-separated list of values ​​(e.g., `50,100,200`).
- The `--candidate-step` parameter applies only when using `--algo pdss`.
- `--gpus 0,1,2` overrides automatic GPU detection (via the `nvidia-smi -L` command); if this parameter is omitted, all detected GPUs will be used.

Results are saved in `<outdir>/<algo>/<series>/seglen_<L>/`. Data for the best length is copied to `<outdir>/<algo>/<series>/best_seglen_<L>/`, and the file `<outdir>/<algo>/<series>/summary.json` contains information on the predicted and actual execution times, as well as an estimate of the area between the profiles for each length.

# Citation

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

# Acknowledgement

This work was financially supported by the Russian Science Foundation (grant no. 23-21-00465).