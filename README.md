# Superconcurrent Sklearn Trainer
The Superparallel Sklearn Trainer is a pure-python, distributed training framework that requires a shared file system for operation. It features yaml-based hyperparameter setup and a manager that sets up a passive worker arguments structure and handles cleanup operations that allows for flexible startup and rerun of worker processes as needed. Drawing inspiration from the MapReduce approach, this trainer emphasizes readability, customization, and persistent execution, making massive hyperparameter search exploration more manageable and transparent.

The object-oriented argument handling and results saving allows for easy addition of attributes for analysis after runtime in a notebook environment.

## For different datasets:
- Replace data and metadata files in `data` directory.
- Change filter function in `scripts/utils.py` and filter configurations in `configs/config.yaml` to match your filtering use case.
- Change any process calls to match any new/changed pipelines and split filter configurations
    - This is currently set up for two class, but could be multiclass by simply adjusting pipe_configurations and the y_col arguments.
    - Main changes will be in train.py manager call.
- Change your desired model pipelines in `configs/pipe_configs.yaml`.
- 'Condition' and 'Run' are used to target the y-col in metadata and indexing in the two datasets. New application should change these target columns in `configs/config.yaml`.

## Currently storage intensive
- If this is an issue, the model doesn't need to be saved. Models can be recreated from their ModelConfig objects given a split arg at any time! Disable model saving in `scripts/train.Worker.train_eval_save()`. Note: random processes may need seeding input in `configs/pipe_configs.yaml` for perfect reproducability.

## Starting a Train

### Environment Setup
- Any conda environment can run this process by simply executing manager and running as many worker processes as available.
- Activate your Conda environment.
- Navigate to the `scripts` directory.
- If you're running on Berkeley's Savio, there is good documentation for conda [here](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/software/using-software/using-python-savio/), or, for an exact replication of the environment use `req.txt`.
- If you're not using Berkeley's Savio, `req.txt` is not guaranteed to work, but that and/or environment.yaml should suffice.
- In short, we're going to be making a python 3.11 conda environment, installing mamba as our package manager, and then installing all of the packages needed in `scripts/globals.py`, mainly being `scikit-learn`, `py-xgboost`, `filelock`, and `pyyaml` with mamba.

### Preparing Training (train.py Manager)
- Edit the `train_manager.sh` script to match your `split_names` as defined in `configs/config.yaml`. These are your data splits for training. Note the naming convention in train.py inputs if you'd like to evaluate during runtime with `validate=true`.
- Execute the script using `sbatch train_manager.sh` or an equivalent bash command to set up the model arguments queue and pre-filter the training data for workers.
    - **Hint:** Workers can be initiated as soon as there are enough `ModelConfig` objects in the queue. You will receive a log notification when the manager starts adding these arguments. It's theoretically possible too many workers could pull args faster than the manager can put them in.
- **Hint:** You can monitor the completion of all training steps in `status.log` with `tail -f ../logs/status.log`.
- **Note:** The order that arguments are added to the queue matters in the context of SplitConfig context switches. Workers store then drop their train data, and will be held up by i/o if they consistently need to read in new train data. See `train.Worker.ensure_split_config()` for more info.
- Once "Manager Complete." is logged, proceed launch workers (if you haven't already).

### Running Workers
- Run `train.py` with `work` arg to start a worker (no other args needed, but worker_id may be helpful for logging):
    - A single invocation of `train.py` launches one worker who will continue to train on available arguments until completion or termination. Current slurm script (`train_workers.sh`) launches an array of invocations.
    - All tasks outside of argument retrieval are designed to be handled independently by each worker.
    - Arguments are queued in a random, interleaved order at the pipeline name level, ensuring a balanced exploration across all pipeline types for single split trains. For multiple splits, models may only begin later splits as they progress.
    - Any worker stuck on a single argument for more than an hour will drop that process and move to a new argument (unix feature only). This timeout can be adjusted in `train.py`.

## Example Usage
- To manage pipelines with CV and validation:
  ```
  python train.py manage --pipeline_names lasso enet svc xgb rf nn --split_names A1 B2 --cv True --validate True
  ```
- To manage pipelines without CV nor validation:
  ```
  python train.py manage --pipeline_names lasso enet --split_names train_A1 train_B2 --cv False --validate False
  ```
  **Note:** Both commands would train on `train_A1`, but the first would also require `validate_A1` to be present in the config. See type hints in train.py for more details.

## Collecting Results
- Results can be collected by creating a `DistributedArgHandler` from the with `globals.SCORES_SAVE_QUEUE_PATH` as its location. `DistrutedArgHandler.get_all()` should be a read-only operation that lists all model_config objects in list form, but I reccomend you create a copy of your results to avoid the risk of losing your train session's progress.
- The objects being stored in the results queue are utils.ModelConfig objects, and the available parameters vary depending on your cv and validate arguments.

## Possible Additions
- Make sure all hyperparameters can still be searched in the case of early stop:
    - Manager could have a way of knowing it's status in adding arguments to the queue.
    - Workers could hold their lock until completion, only then deleting their model_config object from the arg queue.
