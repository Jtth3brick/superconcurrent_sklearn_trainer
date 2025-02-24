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
    return utils.DistributedArgHandler(location=MODEL_CONFIGS_PATH, lock_timeout=timeout, scan_limit=500)

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

            try:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(MAX_TRAIN_TIME)
                self.train_eval_save(model_config)
                signal.alarm(0)
            except TimeoutError:
                self.logger.error(f"TimeoutError: eval_assemble_save did not complete within 60 minutes for config={str(model_config)}")
            except Exception as e:
                self.logger.error(f"Error occurred in eval_assemble_save: {str(e)}\n\tconfig={str(model_config)}")
            finally:
                # Explicitly clear model_config to help garbage collection
                model_config = None
                gc.collect()
        
    def train_eval_save(self, model_config: utils.ModelConfig, extract_features=True, save_model=True) -> None:
        self.logger.info(f"Training model:\n\t{model_config}")

        try:
            # Convert to clean names for model training
            X_train, y_train = self.split_config.X_train.copy(), self.split_config.y_train.copy()
            if self.split_config.X_val is not None:
                X_val, y_val = self.split_config.X_val.copy(), self.split_config.y_val.copy()
            else:
                X_val, y_val = None, None

            # Check if we should do CV
            if self.split_config.cv_indexes: 
                model_config.cv_scores = self.get_cv_scores(model_config)

            # train full model
            pipe = model_config.get_empty_pipe()
            pipe.fit(X_train, y_train)

            # save fitted pipe
            if save_model:
                model_file_path = MODELS_DIR / f"{model_config.config_hash}.model.pkl"
                with open(model_file_path, "wb") as file:
                    pickle.dump(pipe, file)
            
            # extract top features
            if extract_features:
                model_config.top_k_features = utils.extract_topK_features(pipe)
                self.logger.info(f"Top features extracted: {list(model_config.top_k_features.keys())}")

            # Check if we want validate scores
            if X_val is not None:
                y_pred_proba = pipe.predict_proba(X_val)
                model_config.validate_score = utils.get_score(y_true=y_val, y_pred_proba=y_pred_proba)

            # Save scored utils.ModelConfig
            self.save_queue.put(model_config, obj_name=model_config.config_hash, group_name=model_config.split_name)
        finally:
            # Explicitly clear references to free memory
            pipe = None
            X_train = None
            y_train = None
            X_val = None
            y_val = None
            gc.collect()
    
    def get_cv_scores(self, model_config: utils.ModelConfig) -> List[float]:
        scores = []
        X_train_original, y_train_original = self.split_config.X_train.copy(), self.split_config.y_train.copy()
        
        try:
            for i, (cv_train_indices, cv_test_indices) in enumerate(self.split_config.cv_indexes):
                pipe = model_config.get_empty_pipe()
                
                X_train_fold = X_train_original.loc[cv_train_indices]
                y_train_fold = y_train_original.loc[cv_train_indices]
                X_val_fold = X_train_original.loc[cv_test_indices] 
                y_val_fold = y_train_original.loc[cv_test_indices]
                
                pipe.fit(X_train_fold, y_train_fold)
                y_pred_proba = pipe.predict_proba(X_val_fold)
                score = utils.get_score(y_true=y_val_fold, y_pred_proba=y_pred_proba, trained_classes=pipe.classes_)
                scores.append(score)
                
                # Clear fold-specific objects
                pipe = None
                X_train_fold = None
                y_train_fold = None
                X_val_fold = None
                y_val_fold = None
                
                # Run garbage collection periodically during CV
                if i % 2 == 0:  # Every other fold
                    gc.collect()
                    
            return scores
        finally:
            # Clean up original data copies
            X_train_original = None
            y_train_original = None
            gc.collect()

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
                
                # Explicitly clear the old split_config to help garbage collection
                self.split_config = None
                gc.collect()
                
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
            self.logger.info(f"Getting {train_split_name} training data...")
            X_train, y_train = utils.get_data(train_split_name)
            self.logger.info(f"Retrieved {train_split_name} train data with shape {X_train.shape}")

            self.logger.info(f"Getting {validate_split_name} validate data...")
            X_val, y_val = utils.get_data(validate_split_name, schema=list(X_train.columns))
            self.logger.info(f"Retrieved {validate_split_name} validate data with shape {X_val.shape}")
        else:
            self.logger.info(f"Validation scoring skipped. Getting training data only (split name is {split_name})...")
            X_train, y_train = utils.get_data(split_name)
            self.logger.info(f"Retrieved {split_name} data with shape {X_train.shape}")

            # Indicate that validate should not be done
            X_val, y_val = None, None
        
        # Get indexes for cv splits if cv was indicated
        if self.cv:
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
            # Convert DataFrame indices to run indices before storing
            run_indices = list(X_train.index)
            cv_indexes = []
            for train_idx, test_idx in skf.split(X_train, y_train):
                # Convert DataFrame indices to run names
                train_runs = [run_indices[i] for i in train_idx]
                test_runs = [run_indices[i] for i in test_idx]
                cv_indexes.append((train_runs, test_runs))
        else:
            # Indicate CV should not be done
            cv_indexes = None

        # Create SplitConfig
        split_config = utils.SplitConfig(
            split_name=split_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cv_indexes=cv_indexes
        )

        return split_config
    
    @staticmethod
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