# Video annotator

## Data
TODO: upload the data folder to gdrive

1. Download the data tar file from [here](http://todo.com/todo) and unpack into your folder of choice.
2. Set `DATA_BASE_PATH` to that folder in `videoannotator/config.py`. E.g. `/root/data/videoannotator/`.
3. View [this notebook](data-exploration.ipynb) to understand how the data is organized.

Once unpacked, the data should look like this:
```
├── videoannotator
│   ├── agg
│   │   ├── action.json
│   │   ├── aerial.json
│   │   ├── ...
│   ├── ave
│   │   ├── keys-to-labels.json
│   │   ├── labels-mapped.json
│   │   ├── ...
│   ├── checkpoints
│   │   ├── action.json
│   │   ├── aerial.json
│   │   ├── ...
│   ├── cmp
│   │   ├── action.json
│   │   ├── aerial.json
│   │   ├── ...
│   ├── embeddings.h5
│   ├── queries.json
│   ├── shot-data.csv
│   ├── text-embeddings.json
```

## Environment setup
```shell
conda env create -f conda_env.yml
conda activate videoannotator
```

## Running experiments
- [Experiment 1 notebook](exp1.ipynb)
- [Experiment 2 notebook](exp2.ipynb)

## Citation
TODO
