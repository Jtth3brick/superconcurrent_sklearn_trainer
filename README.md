# Superparallel Sklearn Trainer
The Superparallel Sklearn Trainer is a parallel and distributed training framework that requires a shared file system for operation. It features a running manager that handles cleanup operations, while allowing for the flexible startup and rerun of worker processes as needed. Drawing inspiration from the MapReduce approach, this trainer emphasizes readability, explicitness, and persistent execution, making massive training tasks more manageable and transparent.

## For different datasets:
    - Replace data and metadata files in `data` directory.
    - Change filter function in `scripts/utils.py` and `configs/config.yaml` to match your filtering use case.
    - Change any process calls to match any new/changed pipelines and split filter configurations
        - This is currently set up for two class, but could be multiclass by simply adjusting pipe_configurations and the y_col arguments.
    - Change your desired model pipelines in `configs/pipe_configs.yaml`
    - 'Condition' and 'Run' are used to target the y-col in metadata and indexing in the two datasets. New application should change these target columns in `configs/config.yaml`.

## Doesn't actually need to be on slurm!
    - Any conda environment can run this process by simply executing manager and running as many worker processes as wanted

Currently storage intensive
    - If this is an issue, the model doesn't need to be saved. Models can be recreated from ModelConfig objects easily!

## Starting a Train

### Environment Setup
- Load the Python module (if necessary) and activate your Conda environment.
- Navigate to the `scripts` directory.
- If you're running on Berkeley's Savio, there is good documentation for conda [here](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/software/using-software/using-python-savio/), or, for an exact replication of the environment use `req.txt`.
    - If you're not using Berkeley's Savio, `req.txt` is not guaranteed to work, but that and/or environment.yaml should suffice.
    - In short, we're going to be making a python 3.11 conda environment, installing mamba as our package manager, and then installing all of the packages needed in `scripts/globals.py`, mainly being `scikit-learn`, `py-xgboost`, `filelock`, and `pyyaml` with mamba and `persist-queue` with pip

### Preparing Training (train.py Manager)
- Edit the `train_manager.sh` script to match your `split_names` as defined in `configs/config.yaml`. These are your data splits for training.
- Execute the script using `sbatch train_manager.sh` or an equivalent bash command to set up the model arguments queue and pre-filter the training data for workers.
    - **Hint:** Workers can be initiated as soon as there are enough `ModelConfig` objects in the queue. You will receive a log notification when the manager starts adding these arguments. It's possible too many workers could pull args faster than the manager can put them in (estimated on the scale of 700 workers.) So waiting for manager to finish may be your safest bet. 
- **Hint:** You can monitor the completion of all training steps in `status.log` with `tail -f ../logs/status.log`.
- Once "Manager Complete." is logged, proceed to the next step (if you haven't already).

### Running Workers
- Run `train.py` with `work` argto start a worker:
    - A single invocation of `train.py` launches one worker who will continue to train on available arguments until completion or termination. Current slurm script launches an array of invocations.
    - All tasks outside of argument retrieval are designed to be handled independently by each worker.
    - Workers can be dynamically started or stopped as needed. Note that stopping a worker results in the loss of its current hyperparameter configuration. Future versions could implement ACK functionality from persist-queue to ensure comprehensive search.
    - Arguments are queued in a random, interleaved order at the pipeline name level, ensuring a balanced exploration across all pipeline types for single split trains. For multiple splits, models may only begin later splits as they progress.
    - Any worker stuck on a single argument for more than an hour will drop that process and move to a new argument. This timeout can be adjusted in `train.py`.

## Example Usage
- To manage pipelines with CV and validation:
  ```
  python train.py manage --pipeline_names lasso enet svc xgb rf nn --split_names A1 B2 --cv True --validate True
  ```
- To manage pipelines without CV nor validation:
  ```
  python train.py manage --pipeline_names lasso enet --split_names train_A1 train_B2 --cv False --validate False
  ```
  **Note:** Both commands would train on `train_A1`, but the first would also require `validate_A1` to be present in the config.
