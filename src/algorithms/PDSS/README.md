# PDSS (Parallel Discovery of Semantic Snippets)

This repository is related to the PDSS (Parallel Discovery of Summary Snippets) algorithm that automates summary snippet discovery in large time series with a graphics processor. PDSS is authored by Andrey Goglachev (goglachevai@susu.ru) and Mikhail Zymbler (mzym@susu.ru), South Ural State University, Chelyabinsk, Russia. The repository contains the PDSS' source code (in C, CUDA).

## Usage

```
PDSS <input_file> <output_dir> <time_series_length> <segment_length> <num_snippets> [candidate_step]
```

| Parameter             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `input_file`          | Text file containing the time series (numbers separated by spaces/newlines) |
| `output_dir`          | Directory for results (created automatically)                               |
| `time_series_length`  | Length of the input time series                                             |
| `segment_length`      | Length of the snippet                                                       |
| `num_snippets`        | Number of snippets to find                                                  |
| `candidate_step`      | Step size between candidate segments (optional; defaults to `segment_length`, i.e., no overlap) |

Example:


```
PDSS GreatBarbet1F_Jog_1800_200.txt results 1800 200 2 50
```

## Output

The following files are created in `output_dir`:

- `indicies.txt` — the starting index of each found snippet within the original series.
- `snippets.txt` — the values ​​of each snippet.
- `fracs.txt` — the fraction of the series covered by each snippet.
- `labels.txt` — labeling of the input time series.
- `profiles.txt` — the C22dist profile for each found snippet.
