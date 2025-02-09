from globals import *

_loggers = {}

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
            iterator = tqdm(total=total, disable=not report_status, desc="Loading objects")
            
            last_percent = -1
            for entry in it:
                if entry.name.endswith('.pkl'):
                    if limit is not None and len(results) >= limit:
                        break
                    file_path = self.location / entry.name
                    try:
                        with open(file_path, 'rb') as f:
                            obj = pickle.load(f)
                        results.append(obj)
                        
                        # Update progress bar only when percentage changes
                        current_percent = math.floor((len(results) / total) * 100)
                        if report_status and current_percent > last_percent:
                            iterator.update(current_percent - last_percent)
                            iterator.set_description(f"Loaded {len(results)} objects ({current_percent}%)")
                            last_percent = current_percent
                    except:
                        # If we fail to read a file, just skip it
                        continue

        if report_status:
            iterator.close()
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
    validate_data: pd.DataFrame

@dataclass
class SplitConfig:
    """
    Contains data to train a model on.

    Attributes:
        split_name (str): Identifier for the data split.
        train_query (str): SQL query to get training data.
        validate_query (str): SQL query to get validation data.
        column_order (List[str]): List of column names in order to match training schema.
        cv_indexes (List[Tuple[np.ndarray, np.ndarray]]): Contains the splits for cross validation.
        - None iff cv is False
    """
    split_name: str
    train_query: str
    validate_query: str
    column_order: List[str]
    cv_indexes: List[Tuple[np.ndarray, np.ndarray]]

def get_data(query: str, schema: Optional[List[str]] = None) -> Union[Tuple[np.ndarray, List[str]], np.ndarray]:
    """
    Get data from database using a query that returns run IDs, optionally fitting to a schema.
    
    Parameters
    ----------
    query : str
        SQL query that returns a list of run IDs (like those in queries.yaml)
    schema : Optional[List[str]]
        Optional list of feature names to fit the returned data to.
        If not provided, returns both data and the schema.
    
    Returns
    -------
    If schema provided:
        np.ndarray: Data matrix with rows matching query runs and columns matching schema
    If no schema:
        Tuple[np.ndarray, List[str]]: Data matrix and list of feature names
    """
    # Get run IDs from query
    runs_df = load_dataframe(query)
    runs = runs_df['run'].tolist()
    
    if not runs:
        return np.array([]) if schema else (np.array([]), [])
        
    # Build placeholder list for SQL
    runs_placeholder = ", ".join(f"'{run}'" for run in runs)
    
    # Get genomic features
    features_query = f"""
        SELECT r.run, t.taxon_name, g.rpm
        FROM runs r
        LEFT JOIN genomic_sequence_rpm g ON r.run = g.run
        LEFT JOIN taxa t ON g.taxon_id = t.taxon_id
        WHERE r.run IN ({runs_placeholder})
    """
    features_df = load_dataframe(features_query)
    
    # Pivot features into matrix
    if not features_df.empty and 'taxon_name' in features_df.columns:
        pivot_df = features_df.pivot_table(
            index='run',
            columns='taxon_name',
            values='rpm',
            aggfunc='mean',
            fill_value=0
        )
    else:
        return np.array([]) if schema else (np.array([]), [])

    # If schema provided, reindex to match
    if schema is not None:
        missing_cols = set(schema) - set(pivot_df.columns)
        for col in missing_cols:
            pivot_df[col] = 0
        pivot_df = pivot_df[schema]
        return pivot_df.to_numpy()
    
    # Otherwise return data and feature names
    return pivot_df.to_numpy(), pivot_df.columns.tolist()


def get_logger(context: str = None):
    """
    Retrieve or initialize a logger with the given context.
    Logs are written to LOGS_DIR/status.log.
    """
    global _loggers
    log_file_name = "status.log"
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    if context not in _loggers:
        logger = logging.getLogger(context if context else __name__)
        logger.setLevel(logging.INFO)
        # Avoid adding duplicate handlers
        if log_file_name not in {getattr(handler, 'baseFilename', None) for handler in logger.handlers}:
            handler = logging.FileHandler(LOGS_DIR / log_file_name)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(f'%(asctime)s - {context if context else "general"} - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        _loggers[context] = logger
    return _loggers[context]

# --- MACHINE LEARNING UTILITIES ---

@dataclass
class ModelConfig:
    """
    Contains configuration required for training and evaluating a model.
    
    Attributes:
        split_name (str): Identifier for the data split.
        pipeline_name (str): Name of the training pipeline (e.g., 'enet', 'rf').
        _pipeline_struct (Pipeline): The pipeline structure (preprocessing/modeling steps).
        _pipeline_hyperparameters (Dict[str, Any]): Hyperparameters for the pipeline.
        cv_scores (List[float]): Cross-validation AUC scores.
        validate_score (float): Optional validation AUC score.
        config_hash (str): Unique identifier generated based on configuration.
        top_k_features (Dict[str, float]): Top k features and their importance scores.
    """
    split_name: str
    pipeline_name: str
    _pipeline_struct: 'Pipeline'
    _pipeline_hyperparameters: dict
    cv_scores: list = field(default_factory=list, repr=False)
    validate_score: float = field(default=None, repr=False)
    config_hash: str = field(default=None, repr=False)
    top_k_features: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.config_hash = self.create_hash()

    def create_hash(self) -> str:
        hyperparams_str = str(sorted(self._pipeline_hyperparameters.items()))
        unique_str = f"{hyperparams_str}{self.split_name}"
        return hashlib.sha256(unique_str.encode()).hexdigest()
    
    def get_empty_pipe(self) -> 'Pipeline':
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
                f"  Top Features: {list(self.top_k_features.keys())}\n"
                f")")

def stratified_cv(df: 'pd.DataFrame', seed: int, n_splits: int):
    """
    Generates stratified cross-validation splits based on the 'Condition' column.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'Condition' column.
        seed (int): Random seed for reproducibility.
        n_splits (int): Number of CV splits.
        
    Returns:
        list: A list of tuples containing train and test indices.
        
    Raises:
        ValueError: If 'Condition' column is missing.
    """
    if 'Condition' not in df.columns:
        raise ValueError(f"'Condition' column is not in the input DataFrame.")
    
    labels = df['Condition'].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, test_idx in skf.split(df, labels):
        splits.append((df.index[train_idx], df.index[test_idx]))
    return splits

def prep_search(pipeline_config: dict):
    """
    Prepares a pipeline and a hyperparameter grid for a model search.
    
    Args:
        pipeline_config (dict): Dictionary containing pipeline steps and configurations.
        
    Returns:
        tuple: (Pipeline, List of parameter grids)
    """
    def import_class(module_path: str):
        module_name, class_name = module_path.rsplit(".", 1)
        module = import_module(module_name)
        return getattr(module, class_name)

    step_names = [list(step.keys())[0] for step in pipeline_config['steps']]
    step_configs_lists = [list(step.values())[0] for step in pipeline_config['steps']]
    pipeline = Pipeline([(step_name, 'passthrough') for step_name in step_names])
    param_grids = []

    for step_configs in product(*step_configs_lists):
        param_grid = {}
        for step_name, step_config in zip(step_names, step_configs):
            func = import_class(step_config['function'])
            args = step_config.get('args', {})
            param_grid[f"{step_name}"] = [func(**args)]
            for hp_name, hp_values in step_config.get('hyperparams', {}).items():
                param_grid[f"{step_name}__{hp_name}"] = hp_values
        param_grids.append(param_grid)

    return pipeline, param_grids

def get_all_configurations(grid: list):
    """
    Generates all possible configurations from a list of parameter grids.
    
    Args:
        grid (list): List of parameter grids.
        
    Returns:
        list: All possible configuration dictionaries.
    """
    all_configurations = []
    for param_grid in grid:
        keys = param_grid.keys()
        values = []
        for key, lst in param_grid.items():
            if isinstance(lst[0], type):
                values.append([v() for v in lst])
            else:
                values.append(lst)
        for combination in product(*values):
            configuration = dict(zip(keys, combination))
            all_configurations.append(configuration)
    return all_configurations

def get_score(y_true, y_pred_proba, trained_classes):
    """
    Calculates the weighted AUC score for a multi-class classification problem.
    
    Args:
        y_true (np.array): True labels.
        y_pred_proba (np.array): Predicted probabilities.
        trained_classes (list): The classes the model was trained on.
        
    Returns:
        float: The weighted AUC score.
    """
    logger = get_logger('utils.py -- get_score')
    weighted_auc = 0.0
    total_samples = len(y_true)
    for i in np.unique(y_true):
        if i not in trained_classes:
            logger.warning(f"Class {i} in y_true is not in trained_classes")
            continue
        y_true_binary = (y_true == i).astype(int)
        class_idx = trained_classes.tolist().index(i)
        y_pred_proba_class = y_pred_proba[:, class_idx]
        auc = roc_auc_score(y_true_binary, y_pred_proba_class)
        weight = np.sum(y_true == i) / total_samples
        weighted_auc += auc * weight
    return weighted_auc
