from globals import *
import utils
from utils import ModelConfig, SplitConfig

logging.basicConfig(level=logging.INFO)
LOGGER = utils.get_logger('train.py')

# Move on from hyperparam config after 1 hour
MAX_TRAIN_TIME = 60*60 

# Argument handling constants (through persistant, file-based queue)
MAX_ARG_RETRIEVAL_ATTEMPTS = 3
MODEL_CONFIGS_PATH = WORKING_DATA_DIR / 'model_config_queue'

random.seed(SEED)

"""
For timeout interruption
"""
class TimeoutError(Exception):
    pass
def handler(signum, frame):
    raise TimeoutError()

def get_split_args_path(split_name):
    """
    Given a split name, this function returns a path to the split args.
    """
    return WORKING_DATA_DIR / f'{split_name}_split_args.pkl'

def get_untrained_model_config_queue(timeout):
    """
    Initializes a queue access object.
    """
    return utils.DistributedArgHandler(location=MODEL_CONFIGS_PATH, lock_timeout=timeout, scan_limit=1)

class Worker:
    """
    A Worker class responsible for processing machine learning models. 
    Each worker instance handles a specific task assigned to it by retrieving 
    model configurations from a shared queue, ensuring the appropriate data splits 
    are loaded, and executing model training within specified time limits.

    Attributes:
        worker_id (int): Unique identifier for the worker.
        split_config (SplitConfig): Configuration that contains information about 
                                    how the data is split.
        logger (Logger): Logger instance for logging info, errors, and other relevant events.

    Methods:
        work(): Primary method that continuously retrieves and processes model configurations.
        train_eval_save(model_config): Trains a model and saves it along with its cv scores in results queue.
        ensure_split_config(model_config): Ensures that the correct data split configuration 
                                           is loaded based on the model configuration.
        get_model_config(): Retrieves a new model configuration from the queue.
    """
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.logger = utils.get_logger(f'train worker #{worker_id}')
        self.split_config: SplitConfig | None = None
        self.model_config_queue = get_untrained_model_config_queue(timeout=10) # 1 min timeout for workers
        self.save_queue = utils.DistributedArgHandler(SCORES_SAVE_QUEUE_PATH)

    def work(self):
        self.logger.info(f"Python active")

        while True:
            model_config = self.get_model_config()
            self.ensure_split_config(model_config)

            # Start a timer. If the function does not complete within 60 minutes, 
            # it will raise a TimeoutError skipping that hyperparameter configuration for all remaining batches.
            try:
                signal.alarm(MAX_TRAIN_TIME) # Skip hyperparam config after MAX_TRAIN_TIME
                self.train_eval_save(model_config)
                # Disable the alarm
                signal.alarm(0)
            except TimeoutError:
                self.logger.error(f"TimeoutError: eval_assemble_save did not complete within 60 minutes for config={str(model_config)}")
                continue
            except Exception as e:
                self.logger.error(f"Error occurred in eval_assemble_save: {str(e)}\n\tconfig={str(model_config)}")
    
    def train_eval_save(self, model_config: utils.ModelConfig) -> None:
        self.logger.info(f"Training model:\n\t{model_config}")

        # Check if we should do CV
        if self.split_config.cv_indexes: 
            model_config.cv_scores = self.get_cv_scores(model_config)

        # train full model
        X, y = self.split_config.train_data, self.split_config.train_metadata[Y_COL_NAME]
        pipe = model_config.get_empty_pipe()
        pipe.fit(X.copy(), y.copy())

        # save fitted pipe
        model_file_path = MODELS_DIR / f"{model_config.config_hash}.model.pkl"
        with open(model_file_path, "wb") as file:
            pickle.dump(pipe, file)

        # Check if we want validate scores
        if self.split_config.validate_data is not None:
            X, y = self.split_config.validate_data, self.split_config.validate_metadata[Y_COL_NAME]
            y_pred_proba = pipe.predict_proba(X)
            model_config.validate_score = utils.get_score(y_true=y, y_pred_proba=y_pred_proba, trained_classes=pipe.classes_)

        # Save scored utils.ModelConfig
        self.save_queue.put(model_config, obj_name=model_config.config_hash, group_name=model_config.split_name)
    
    def get_cv_scores(self, model_config: utils.ModelConfig) -> List[float]:
        X, y = self.split_config.train_data, self.split_config.train_metadata[Y_COL_NAME]

        scores = []
        for cv_split in self.split_config.cv_indexes:
            pipe = model_config.get_empty_pipe()
            cv_train_indices, cv_test_indices = cv_split

            X_train, X_test = X.loc[cv_train_indices], X.loc[cv_test_indices]
            y_train, y_test = y.loc[cv_train_indices], y.loc[cv_test_indices]
            pipe.fit(X_train.copy(), y_train.copy())
            y_pred_proba = pipe.predict_proba(X_test)
            score = utils.get_score(y_true=y_test, y_pred_proba=y_pred_proba, trained_classes=pipe.classes_)
            scores.append(score)
        return scores

    def ensure_split_config(self, model_config: utils.ModelConfig):
        """
        Checks if in-memory split config is correct for the given model config and swaps if not.
        """
        needed_split_name = model_config.split_name
        cur_data_split_name = None
        if self.split_config is not None:
            cur_data_split_name = self.split_config.split_name

        if model_config.split_name != cur_data_split_name:
            self.logger.info(f"Switching in memory train data from {cur_data_split_name} to {needed_split_name}.")
            split_config_path = get_split_args_path(needed_split_name)
            with open(split_config_path, 'rb') as f:
                self.split_config = pickle.load(f)

    def get_model_config(self) -> utils.ModelConfig:
        """
        Helper function used by workers to get a new model argument.

        Uses the built-in retry mechanism of DistributedArgHandler.
        On failure, the worker will exit.
        """
        model_config = self.model_config_queue.get()
        if model_config is None:
            self.logger.error(f"Failed to load training argument after {self.model_config_queue.retry_limit} attempts. Estimated number of args remaining: {self.model_config_queue.count_pkl_files()}")
            sys.exit()
        return model_config
class Manager:
    """
    Manages the preparation and distribution of model configurations and data splits.
    Responsible for initializing directories, generating necessary configurations, and 
    queueing them for the workers to process.

    Attributes:
        pipeline_names (list of str): Names of the pipelines to handle.
        split_names (list of str): Names of the data splits to be processed.
        logger (Logger): Logger for tracking the manager's operations.

    Methods:
        manage(): Orchestrates the generation and queuing of model configurations.
        generate_model_configs(pipeline_names, split_name): Generates and returns a list of model configurations.
        generate_split_config(split_name): Generates and returns a split configuration based on the split name.
        prep_directories(): Prepares necessary directories and handles initial data setup.
    """
    def __init__(self, pipeline_names, split_names, cv, validate):
        Manager.prep_directories()
        self.pipeline_names = pipeline_names
        self.split_names = split_names
        self.logger = utils.get_logger(f'train manager')
        self.cv = cv
        self.validate = validate
        self.model_config_queue = get_untrained_model_config_queue(timeout=60*60) # 1 hour timeout for workers

    def manage(self):
        """
        Makes a SplitConfig for each pipeline name in pipeline_names and saves
        then starts adding model args to untrained_model_config_queue.
        """
        added_args = 0
        for i, split_name in enumerate(self.split_names):
            # Generate SplitConfig, get save path, and save
            split_config = self.generate_split_config(split_name)
            split_args_path = get_split_args_path(split_name)
            with open(split_args_path, 'wb') as f:
                    pickle.dump(split_config, f)
            self.logger.info(f"Successfully saved {split_name} SplitConfig.")

            # Generate utils.ModelConfigs and add to queue. We do this second so workers can dequeue items and start working.
            model_configs = self.generate_model_configs(split_name)
            self.logger.info(f"Queueing model args for split {split_name}. Workers may start booting up after some lag...")
            for model_arg in model_configs:
                self.model_config_queue.put(model_arg, obj_name=model_arg.config_hash, group_name=model_arg.split_name)
                added_args += 1
                if added_args % 5000 == 0:
                    self.logger.info(f"Adding {added_args} total arguments. On split {i+1} of {len(self.split_names)+1} total splits. Current estimated size: {self.model_config_queue.count_pkl_files()}")
            self.logger.info(f"Model args for split {split_name} successfully added to queue.")

        self.logger.info(f"Manager Completed.")

    def generate_model_configs(self, split_name: str) -> List[utils.ModelConfig]:
        """
        Generates a list of utils.ModelConfig instances for various pipeline configurations.
        
        This function reads pipeline configurations from a YAML file, prepares pipeline structures and 
        parameter grids, and then creates utils.ModelConfig objects for each combination of parameters. It ensures 
        that the configurations are interlaced for balanced training across different pipeline types.
        
        Returns:
            list: A list of interlaced utils.ModelConfig instances ready for model training.
        """

        # Load pipeline configurations
        self.logger.info("Loading pipeline configurations.")
        with open(PIPE_CONFIG_PATH, 'r') as f:
            pipe_configs =  yaml.safe_load(f)
        pipe_structs = {} # workers will use a variety of pipeline types for varying number of layers, etc
        nested_pipeline_args = [] # intermediate list, interlaced later
        for pipeline_name in self.pipeline_names:
            pipe_config = pipe_configs[pipeline_name]
            pipe_struct, param_grid = utils.prep_search(pipe_config)

            # Shuffle configs
            pipe_hyperparams_lst = utils.get_all_configurations(param_grid)
            random.shuffle(pipe_hyperparams_lst)

            # Creating a list of utils.ModelConfig instances for this specific pipeline type.
            model_configs_lst = [utils.ModelConfig(
                split_name=split_name,
                pipeline_name=pipeline_name,
                _pipeline_struct=pipe_struct,
                _pipeline_hyperparameters=pipe_hyperparams,
                cv_scores=[],
                config_hash=None
            )
            for pipe_hyperparams in pipe_hyperparams_lst]

            # Adding as a nested list so pipeline types can be interlaced and trained evenly.
            nested_pipeline_args.append(model_configs_lst)
        
        # Interlace the different pipelines so they are trained in a balanced order in case of early exit
        interlaced_model_configs = [item for sublist in zip_longest(*nested_pipeline_args) for item in sublist if item]
        self.logger.info(f"size of search: {len(interlaced_model_configs)}")

        return interlaced_model_configs

    def generate_split_config(self, split_name):

        # Getting data sets
        if self.validate:
            # validate requires special naming scheme
            train_split_name = f"train_{split_name}"
            validate_split_name = f"validate_{split_name}"

            # load datasets
            self.logger.info(f"Filtering data on {train_split_name}...")
            train_metadata, train_data = utils.filter_data(train_split_name)

            self.logger.info(f"Filtering validation data on {validate_split_name}...")
            validate_metadata, validate_data = utils.filter_data(validate_split_name)
        else:
            self.logger.info(f"Validation scoring skipped. Filtering data only (split name is {split_name})...")
            train_metadata, train_data = utils.filter_data(split_name)

            # Indicate that validate should not be done
            validate_metadata, validate_data = None, None
        
        # Get indexes for cv splits if cv was indicated
        if self.cv:
            cv_indexes = utils.stratified_cv(train_metadata, SEED, N_SPLITS)
        else:
            # Indicate CV should not be done
            cv_indexes = None

        # Create SplitConfig
        split_config = SplitConfig(
            split_name=split_name,
            train_data=train_data,
            train_metadata=train_metadata,
            validate_data=validate_data,
            validate_metadata=validate_metadata,
            cv_indexes=cv_indexes
        )

        return split_config
    
    def prep_directories():
        # Delete model arg queue if it exists
        if MODEL_CONFIGS_PATH.exists():
            shutil.rmtree(MODEL_CONFIGS_PATH)

        # Delete scores queue if it exists
        if SCORES_SAVE_QUEUE_PATH.exists():
            shutil.rmtree(SCORES_SAVE_QUEUE_PATH)

        # Create necessary directories if they don't exist
        for dir_path in [RESULTS_DIR, MODELS_DIR, WORKING_DATA_DIR, LOGS_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # Unzip data.csv
        zip_data_path = DATA_DIR / 'data.csv.zip'
        with zipfile.ZipFile(zip_data_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to manage or work on machine learning pipelines.")
    parser.add_argument("role", choices=['manage', 'work'], help="Role of the script: 'manage' or 'work'")
    parser.add_argument("--pipeline_names", nargs='+', help="List of pipeline names in pipeline configs. Example: --pipeline_names pipeline1 pipeline2")
    parser.add_argument("--cv", type=bool, help="Whether to include the model's CV score.")
    parser.add_argument("--validate", type=bool, help="Include validation score; requires both 'train_' and 'validate_' splits in config.")
    parser.add_argument("--split_names", nargs='+', help="List of splits to train on. Example: '--split_names train_A1 Train_B2' if --validate is False, else '--split_names A1 B2'")
    parser.add_argument("--worker_id", type=int, help="Optional worker ID for distinguishing workers")

    args = parser.parse_args()

    if args.role == 'manage':
        try:
            manager = Manager(pipeline_names=args.pipeline_names, split_names=args.split_names, cv=args.cv, validate=args.validate)
            manager.manage()
        except Exception as e:
            LOGGER.error(f"Manager encountered an issue:\n" + ''.join(traceback.format_exception(None, e, e.__traceback__)))
            sys.exit(1)
    elif args.role == 'work':
        try:
            worker = Worker(args.worker_id)
            worker.work()
        except Exception as e:
            LOGGER.error(f"Worker {args.worker_id} encountered an issue:\n" + ''.join(traceback.format_exception(None, e, e.__traceback__)))
            sys.exit(1)

# Example usage:
# To manage pipelines with CV and validation:
# python train.py manage --pipeline_names lasso enet svc xgb rf nn --split_names A1 B2 --cv True --validate True
# To work with without CV nor validation:
# python train.py manage --pipeline_names lasso enet --split_names train_A1 train_B2 --cv False --validate False
## Both would train on train_A1, but the first would also require 'validate_A1' present in config, etc.