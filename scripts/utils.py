from globals import *

_loggers = {}

def get_logger(context=None):
    global _loggers

    # Determine log file name
    log_file_name = "status.log"

    # Create the logging directory if it doesn't exist
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    if context not in _loggers:
        _logger = logging.getLogger(context if context else __name__)
        _logger.setLevel(logging.INFO)

        if log_file_name not in {handler.baseFilename for handler in _logger.handlers}:
            # Create a file handler
            handler = logging.FileHandler(LOGS_DIR / log_file_name)
            handler.setLevel(logging.INFO)

            # Create a logging format
            formatter = logging.Formatter(f'%(asctime)s - {context if context else "general"} - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)

            # Add the handlers to the logger
            _logger.addHandler(handler)

        _loggers[context] = _logger

    return _loggers[context]

class DistributedArgHandler:
    """
    A distributed argument handler for managing ModelConfig objects across multiple processes.
    
    This class provides a file-based queue-like interface for storing and retrieving
    ModelConfig objects in a distributed environment. It's designed to work with
    the training pipeline, allowing multiple workers to safely access and process
    model configurations.

    The handler uses file locking to ensure thread-safe and multi-environment safe
    access on a shared file system, making it suitable for use with job schedulers
    like SLURM.

    It also includes bulk operations that are not concurrent-safe but provide
    faster processing for single-threaded scenarios.
    """

    def __init__(self, location: str, lock_timeout: int = 60, retry_limit: int = 5, scan_limit: int = 100):
        """
        Initialize the DistributedArgHandler.

        Args:
            location (str): Path to the directory where ModelConfig objects will be stored.
            lock_timeout (int): Maximum time (in seconds) to wait for a file lock.
            retry_limit (int): Number of times to retry getting an object before giving up.
            scan_limit (int): Maximum number of files to scan in the directory at once.
        """
        self.location = Path(location)
        self.lock_timeout = lock_timeout
        self.retry_limit = retry_limit
        self.scan_limit = scan_limit
        self.location.mkdir(parents=True, exist_ok=True)

    def put(self, obj: Any, obj_name: str, group_name: str = ""):
        """
        Store a ModelConfig object in the distributed storage.

        This method pickles the object and saves it to a file. The filename includes
        the group_name (if provided) to allow for implicit grouping of related configs.

        Args:
            obj (Any): The ModelConfig object to store.
            obj_name (str): A unique name for this object.
            group_name (str, optional): A group identifier, useful for related configs.
        """
        file_name = f"{group_name}_{obj_name}" if group_name else obj_name
        file_path = self.location / f"{file_name}.pkl"
        lock_path = str(file_path) + ".lock"
        
        with FileLock(lock_path, timeout=self.lock_timeout):
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
        
        # Explicitly remove the lock file after it's released
        if os.path.exists(lock_path):
            os.remove(lock_path)


    def get(self) -> Optional[Any]:
        """
        Retrieve and remove a ModelConfig object from the distributed storage.

        This method scans the directory for available files, chooses one randomly,
        and attempts to lock, read, and delete it. If it fails (e.g., due to 
        another process accessing the same file), it will retry up to retry_limit times.

        Returns:
            Optional[Any]: A ModelConfig object if successful, None if no objects are 
                           available or if all retrieval attempts fail.
        """
        for _ in range(self.retry_limit):
            available_files = self._scan_dir()
            if not available_files:
                return None
            
            chosen_file = random.choice(available_files)
            file_path = self.location / chosen_file
            lock_path = str(file_path) + ".lock"
            
            try:
                with FileLock(lock_path, timeout=self.lock_timeout):
                    if file_path.exists():
                        with open(file_path, 'rb') as f:
                            obj = pickle.load(f)
                        file_path.unlink()  # Remove the file after successful retrieval
                        return obj
            except:
                continue  # If we fail to acquire the lock or read the file, try another
        
        return None  # If we've exhausted our retries, return None

    def bulk_put(self, objs: List[Dict[str, Any]]):
        """
        Store multiple ModelConfig objects in the distributed storage.

        NOT CONCURRENT SAFE
        NO TIME COMPLEXITY GUARANTEES

        This method quickly writes multiple objects without using locks,
        assuming no concurrent access.

        Args:
            objs (List[Dict[str, Any]]): A list of dictionaries, each containing:
                - 'obj': The ModelConfig object to store
                - 'obj_name': A unique name for this object
                - 'group_name': (optional) A group identifier
        """
        for item in objs:
            obj = item['obj']
            obj_name = item['obj_name']
            group_name = item.get('group_name', '')
            
            file_name = f"{group_name}_{obj_name}" if group_name else obj_name
            file_path = self.location / f"{file_name}.pkl"
            
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)

    def get_all(self, limit: int = None, report_status=False) -> List[Any]:
        """
        Retrieve (but do not remove) multiple ModelConfig objects from the distributed storage.
        NOT CONCURRENT SAFE
        NO UNIQUENESS GUARANTEES
        NO TIME COMPLEXITY GUARANTEES
        This method quickly reads multiple objects without using locks,
        assuming no concurrent access. It does not modify the storage.
        Args:
        limit (int, optional): Maximum number of objects to retrieve.
        If None, retrieves all available objects.
        report_status (bool, optional): If True, shows a progress bar.
        Returns:
        List[Any]: A list of ModelConfig objects.
        """
        results = []
        
        # Get the total number of .pkl files
        total_files = sum(1 for entry in os.scandir(self.location) if entry.name.endswith('.pkl'))
        
        # Use limit if provided, otherwise use total_files
        total = min(limit, total_files) if limit is not None else total_files

        with os.scandir(self.location) as it:
            # Wrap the iterator with tqdm if report_status is True
            iterator = tqdm(it, total=total, disable=not report_status, desc="Loading objects")
            
            for entry in iterator:
                if entry.name.endswith('.pkl'):
                    if limit is not None and len(results) >= limit:
                        break
                    file_path = self.location / entry.name
                    try:
                        with open(file_path, 'rb') as f:
                            obj = pickle.load(f)
                        results.append(obj)
                    except:
                        # If we fail to read a file, just skip it
                        continue
                    
                    # Update the progress bar description
                    if report_status:
                        iterator.set_description(f"Loaded {len(results)} objects")

        return results

    def _scan_dir(self) -> list:
        """
        Scan the directory for available ModelConfig files.

        This method uses os.scandir for efficient directory scanning, limiting
        the number of files it looks at to scan_limit. It checks for .pkl files
        that don't have an associated .lock file.

        Returns:
            list: A list of available file names, up to scan_limit in length.
        """
        available_files = []
        with os.scandir(self.location) as it:
            for entry in it:
                if len(available_files) >= self.scan_limit:
                    break
                if entry.name.endswith('.pkl') and not Path(str(entry.path) + ".lock").exists():
                    available_files.append(entry.name)
        return available_files
    
    def count_pkl_files(self) -> int:
        """
        Approximate the number of .pkl files in the storage directory.

        This method is READ-ONLY and does not modify any files.
        NO TIME COMPLEXITY GUARANTEES

        Returns:
            int: The number of .pkl files in the storage directory.
        """
        count = 0
        with os.scandir(self.location) as it:
            for entry in it:
                if entry.name.endswith('.pkl'):
                    count += 1
        return count

@dataclass
class ModelConfig:
    """
    Contains everything needed to train one model and view its scores after the fact.
    
    Attributes:
        split_name (str): Identifier for the data split.
        - Filled by manager
        pipeline_name (str): Name of pipeline, e.g. 'enet', 'rf'
        - Filled by manager
        _pipeline_struct (Pipeline): The pipeline structure for preprocessing and modeling. 
        - Filled by manager
        _pipeline_hyperparams (Dict[str, Any]): Dictionary of hyperparameters for the pipeline. This variable must be copied before any fitting.
        - Filled by manager
        config_hash (str): The identifier of the model, specific to hyperparameters and split.
        - Filled automatically after initialization
        cv_scores (List[float]): List of cross-validation AUC scores of the model. Added after fitting and before saving.
        - Filled by worker if cv=true
        validate_score (float): optional AUC score of model applied to validate substitution (see readme)
        - Filled by worker if validate=true
    """
    split_name: str
    pipeline_name: str
    _pipeline_struct: Pipeline
    _pipeline_hyperparameters: Dict[str, Any]
    cv_scores: List[float] = field(default_factory=list, repr=False)
    validate_score: float = field(default=None, repr=False)
    config_hash: str = field(default=None, repr=False)

    def __post_init__(self):
        """
        Post-initialization to ensure the config_hash is generated automatically.
        """
        self.config_hash = self.create_hash()

    def create_hash(self) -> str:
        """
        Generates a SHA-256 hash that uniquely identifies the configuration by combining
        the pipeline_hyperparameters and split_name.
        """
        # Convert the dictionary of hyperparameters to a sorted string to ensure consistent hash generation
        hyperparams_str = str(sorted(self._pipeline_hyperparameters.items()))
        unique_str = f"{hyperparams_str}{self.split_name}"
        return hashlib.sha256(unique_str.encode()).hexdigest()
    
    def get_empty_pipe(self) -> Pipeline:
        pipe_struct = copy.deepcopy(self._pipeline_struct)
        hyper_params = copy.deepcopy(self._pipeline_hyperparameters)

        return pipe_struct.set_params(**hyper_params)

    def __str__(self):
        return (f"ModelConfig(\n"
                f"  Config Hash: {self.config_hash}\n"
                f"  Split Name: {self.split_name}\n"
                f"  Pipeline Name: {self.pipeline_name}\n"
                f"  Pipeline Structure: {self._pipeline_struct}\n"
                f"  Hyperparameters: {self._pipeline_hyperparameters}\n"
                f")")

@dataclass
class SplitConfig:
    """
    Contains data to train a model on.

    Attributes:
        split_name (str): Identifier for the data split.
        train_data (pd.DataFrame): Data to train on.
        train_metadata (pd.DataFrame): Metadata that contains labels.
        cv_indexes (List[Tuple[np.ndarray, np.ndarray]]): Contains the splits for cross validation.
        - None iff cv is False
        evaluate_data (pd.DataFrame): Contains data for evaluate dataset
        - None iff evaluate is False
        evaluate_metadata (pd.DataFrame)
        - None iff evaluate is False
    """
    split_name: str
    train_data: pd.DataFrame
    train_metadata: pd.DataFrame
    validate_data: pd.DataFrame
    validate_metadata: pd.DataFrame
    cv_indexes: List[Tuple[np.ndarray, np.ndarray]]

def filter_data(config_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter and augment the data according to the parameters in config file.
    
    Parameters
    ----------
    config_name : str
        name of config in config.yaml
    
    Returns
    -------
    tuple of two DataFrames
        The first DataFrame contains the filtered metadata.
        The second DataFrame contains the data corresponding to the filtered metadata.
    """
    metadata_path = BASE_DIR / CONFIG['clean_metadata_path']
    data_path = BASE_DIR / CONFIG['clean_data_path']

    # load data and metadata
    filtered_metadata = pd.read_csv(metadata_path)
    data = pd.read_csv(data_path)

    # get filter config from config.yaml
    params = CONFIG[config_name] # parameters for train data
    config = params['config'] # column value pairs that should be filtered on
    include_intervention = params['include_intervention']
    include_remission = params['include_remission']
    include_antibiotics = params['include_antibiotics']
    include_surgery = params['include_surgery']
    include_children = params['include_children']
    include_same_patient_samples = params['include_same_patient_samples']
    combine_uc_and_cd = params['combine_uc_and_cd']

    # Filter the metadata DataFrame based on the specified column and values
    if config:
        for key, values in config.items():
            filtered_metadata = filtered_metadata[filtered_metadata[key].isin(values)].copy()

    # Optionally exclude samples that are under age 18
    if not include_children:
        filtered_metadata = filtered_metadata[filtered_metadata['Age'] >= 18].copy()

    # Optionally exclude samples in remission
    if not include_remission:
        filtered_metadata = filtered_metadata[filtered_metadata['Remission'].astype(bool)==False].copy()

    # Optionally exclude samples that have undergone interventions
    if not include_intervention:
        filtered_metadata = filtered_metadata[filtered_metadata['Intervention'].astype(bool)==False].copy()

    # Optionally exclude samples that have undergone surgery
    if not include_surgery:
        filtered_metadata = filtered_metadata[filtered_metadata['Surgery'].astype(bool)==False].copy()

    # Optionally exclude samples that have undergone antibiotics
    if not include_antibiotics:
        filtered_metadata = filtered_metadata[filtered_metadata['Antibiotics'].astype(bool)==False].copy()

    # Optionally exclude samples from the same patient
    if not include_same_patient_samples:
        filtered_metadata = filtered_metadata.drop_duplicates(subset='Patient ID', keep='first').copy()
    
    # Optionally merge UC and CD Condition column:
    if combine_uc_and_cd:
        filtered_metadata['Condition'] = filtered_metadata['Condition'].replace(2, 1)
        filtered_metadata = filtered_metadata.copy()


    filtered_data = data[data.index.isin(filtered_metadata.index)].copy()

    # Return the filtered data
    return filtered_metadata.set_index(INDEX_COL), filtered_data.set_index(INDEX_COL)

def stratified_cv(df: pd.DataFrame, seed: int, n_splits: int):
    """
    Generates stratified CV splits from the input DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame indexed by accession numbers containing 'Condition' column.
        seed (int): The random seed for reproducibility.
        n_splits (int): The number of CV splits.
        
    Returns:
        list: A list of tuples where each tuple contains two lists representing the train and test splits respectively.
        
    Raises:
        ValueError: If 'Label' column is not found in the input DataFrame.
    """
    
    # Check if 'Label' column exists in 'df'
    if 'Condition' not in df.columns:
        raise ValueError(f"'Condition' column is not in the input DataFrame.")
    
    # Retrieve labels
    labels = df['Condition'].values
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Generate and return the splits
    splits = []
    for train_index, test_index in skf.split(df, labels):
        train, test = df.index[train_index], df.index[test_index]
        splits.append((train, test))
    
    return splits

def prep_search(pipeline_config):

    def import_class(module_path):
        """
        Helper function to initialize a class based on its module path
        """
        module_name, class_name = module_path.rsplit(".", 1)
        module = import_module(module_name)
        return getattr(module, class_name)

    # Get all step names and configs
    step_names = [list(step.keys())[0] for step in pipeline_config['steps']]
    step_configs_lists = [list(step.values())[0] for step in pipeline_config['steps']]

    # Define the pipeline with 'passthrough' as a placeholder for each step
    pipeline = Pipeline([(step_name, 'passthrough') for step_name in step_names])

    # List to store all parameter grid alternatives
    param_grids = []

    # Iterate through all combinations of steps
    for step_configs in product(*step_configs_lists):
        param_grid = {}

        # Iterate through the configuration of each step and initialize the corresponding classes
        for step_name, step_config in zip(step_names, step_configs):
            function = import_class(step_config['function'])
            args = step_config.get('args', {})

            # Add step to the grid
            param_grid[f"{step_name}"] = [function(**args)]

            # Add hyperparameters to the grid
            for hp_name, hp_values in step_config.get('hyperparams', {}).items():
                param_grid[f"{step_name}__{hp_name}"] = hp_values

        # Store the parameter grid
        param_grids.append(param_grid)

    return pipeline, param_grids

def get_all_configurations(grid):
    all_configurations = []
    # Iterate over each parameter grid
    for param_grid in grid:
        keys = param_grid.keys()
        # Get the lists of hyperparameter values for each hyperparameter
        # Keep track of which keys are model objects
        model_keys = []
        values = []
        for key, value in param_grid.items():
            if isinstance(value[0], type):
                model_keys.append(key)
                values.append([v() for v in value])
            else:
                values.append(value)

        # Get all combinations of hyperparameter values
        for combination in product(*values):
            # Create a dict of the combination and add it to the list
            configuration = dict(zip(keys, combination))
            all_configurations.append(configuration)

    return all_configurations


def load_pipeline(hash_id):
    try:
        with open(MODELS_DIR / f'{hash_id}.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pipeline {hash_id}: {e}")
        return None

def get_score(y_true, y_pred_proba, trained_classes):
    """
    Calculate the weighted AUC score for a multi-class classification problem.
    
    Parameters:
    - y_true (numpy array): A 1D array of true labels.
    - y_pred_proba (numpy array): A 2D array where each column contains the predicted probabilities for a class.
    - trained_classes (list): A list of classes on which the model was trained. (Order matters, stored in model params under classes_ attr)
    
    Returns:
    - float: The weighted AUC score.
    """
    
    # Get logger
    logger = get_logger('utils.py -- get_score')

    # Initialize variable to store the weighted AUC
    weighted_auc = 0.0
    
    # Initialize variable to store the total number of samples
    total_samples = len(y_true)
    
    # Loop through each unique class in y_true
    for i in np.unique(y_true):
        # Check if the class exists in trained_classes
        if i not in trained_classes:
            logger.warning(f"Class {i} in y_true is not in trained_classes")
            continue
        
        # Create a binary truth array for the current class
        y_true_binary = (y_true == i).astype(int)
        
        # Extract the predicted probabilities for the current class
        class_idx = trained_classes.tolist().index(i)
        y_pred_proba_class = y_pred_proba[:, class_idx]
        
        # Calculate the AUC for the current class
        auc = roc_auc_score(y_true_binary, y_pred_proba_class)
        
        # Calculate the weight for the current class
        weight = np.sum(y_true == i) / total_samples
        
        # Update the weighted AUC
        weighted_auc += auc * weight
    
    return weighted_auc

