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

    def __init__(self, location: str, lock_timeout: int = 60, retry_limit: int = 5, scan_limit: int = 500):
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
        self._retry_limit = retry_limit
        self.scan_limit = scan_limit
        self.location.mkdir(parents=True, exist_ok=True)

    @property
    def retry_limit(self) -> int:
        """Get the current retry limit."""
        return self._retry_limit

    @retry_limit.setter
    def retry_limit(self, value: int):
        """Set a new retry limit."""
        self._retry_limit = value

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

@dataclass(frozen=True)
class SplitConfig:
    """
    Contains data to train a model on.

    Attributes:
        split_name (str): Identifier for the data split.
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training labels.
        X_val (Optional[pd.DataFrame]): Validation features, if validation is enabled. Assumed to have same columns as X_train.
        y_val (Optional[pd.DataFrame]): Validation labels, if validation is enabled.
        cv_indexes (List[np.ndarray]): Contains the splits for cross validation.
                                     None if cv is False.
    """
    split_name: str
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_val: Optional[pd.DataFrame]
    y_val: Optional[pd.DataFrame]
    cv_indexes: List[np.ndarray]
    
    # Private field for cleaned column names, populated by sanitize_columns()
    _cleaned_cols: List[str] = field(init=False, default_factory=list, repr=False)
    
    @property
    def cleaned_cols(self) -> List[str]:
        """Get the cleaned column names"""
        return self._cleaned_cols
    
    def __post_init__(self) -> None:
        """
        Sanitize the column names based on the original schema and update the DataFrame attributes.
        For each column:
          - Replace non-alphanumeric characters with underscores.
          - Convert to lower-case.
          - Prepend with "f_<index>_".
          - Append the first 4 characters of the SHA-256 hash of the original name.
        """
        cleaned = []
        for i, col in enumerate(self.X_train.columns):
            # Convert to lowercase and replace non-alphanumeric with underscore
            clean_name = re.sub(r'[^a-zA-Z0-9]', '_', col.lower())
            
            # Generate hash of original name
            name_hash = hashlib.sha256(col.encode()).hexdigest()[:4]
            
            # Create final column name
            final_name = f"f_{i}_{clean_name}_{name_hash}"
            cleaned.append(final_name)
            
        object.__setattr__(self, '_cleaned_cols', cleaned)

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
        top_k_features (Optional[Dict[str, float]]): Top k most important features
    """
    # set by manager
    split_name: str
    pipeline_name: str
    _pipeline_struct: Pipeline
    _pipeline_hyperparameters: Dict[str, Any]

    # set by worker after fitting
    model: Optional[Pipeline] = field(default=None, repr=False)
    cv_scores: List[float] = field(default_factory=list, repr=False)
    validate_score: float = field(default=None, repr=False)
    top_k_features: Optional[Dict[str, float]] = None  # top k features as a dict mapping feature name to importance

    _config_hash: str = field(default=None, repr=False)

    @property
    def config_hash(self) -> str:
        """
        Returns the config hash.
        """
        if self._config_hash is None:
            self._config_hash = self.create_hash()
        return self._config_hash

    def create_hash(self) -> str:
        """
        Uses pipeline_hyperparameters and split_name.
        """
        hyperparams_str = str(sorted(self._pipeline_hyperparameters.items()))
        unique_str = f"{hyperparams_str}{self.split_name}"
        return hashlib.sha256(unique_str.encode()).hexdigest()
    
    def get_empty_pipe(self) -> 'Pipeline':
        """Returns a fresh copy of the pipeline with hyperparameters set"""
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
                f"  Top Features: {list(self.top_k_features.keys()) if self.top_k_features is not None else None}\n"
                f")")

def extract_topK_features(pipe, k=10, logger=None):
    """
    Extracts top K features from a fitted pipeline based on feature importance or coefficients.
    
    This function assumes:
      1. The final step of your pipeline is named "model".
      2. The final estimator has either a `feature_importances_` or `coef_` attribute.
      3. You have preserved feature names (for example, by fitting a pandas DataFrame
         or implementing `get_feature_names_out` in earlier steps).

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        A scikit-learn pipeline that has already been fitted.
    k : int, default=10
        Number of top features to extract.

    Returns
    -------
    dict
        Dictionary of the form {feature_name: importance}, sorted by absolute importance,
        containing up to `k` features. Returns an empty dictionary if the function cannot
        determine feature names or importance scores.
    """
    if logger is None:
        logger = type('NullLogger', (), {
            'debug': lambda *args, **kwargs: None,
            'info': lambda *args, **kwargs: None
        })()
    logger.debug("Starting extract_topK_features with k=%d", k)
    
    model = None
    feature_names = None

    # Attempt to locate the final model (assuming its step name is 'model')
    if "model" in pipe.named_steps:
        model = pipe.named_steps["model"]
        logger.debug("Found final model in pipeline: %s", type(model).__name__)
        
        # Attempt to retrieve feature names directly from pipeline
        if hasattr(pipe, "feature_names_in_"):
            feature_names = pipe.feature_names_in_
            logger.debug("Using pipeline.feature_names_in_: %s", feature_names[:5])
        else:
            # Otherwise, check if any intermediate step provides get_feature_names_out()
            for name, step in pipe.named_steps.items():
                if hasattr(step, "get_feature_names_out"):
                    feature_names = step.get_feature_names_out()
                    logger.debug("Using get_feature_names_out() from step '%s': %s", 
                                 name, feature_names[:5])
                    break
    else:
        logger.debug("No final step named 'model' found. Returning empty dict.")
        return {}

    # If we still don't have a model or feature names, return empty
    if model is None or feature_names is None:
        logger.info("No model or feature names found. Returning empty dict.")
        return {}

    # Get importance scores from model
    if hasattr(model, "feature_importances_"):
        logger.debug("Using feature_importances_ from model '%s'.", type(model).__name__)
        importance_scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        logger.debug("Using coef_ from model '%s'.", type(model).__name__)
        coef = model.coef_
        # Use absolute values of coefficients (handle multi-dimensional coef for multi-class)
        importance_scores = abs(coef[0] if coef.ndim > 1 else coef)
    else:
        logger.info("Model '%s' has no feature_importances_ or coef_. Returning empty dict.", 
                    type(model).__name__)
        return {}

    # Create dictionary pairing feature names with importance scores
    feature_importance = dict(zip(feature_names, importance_scores))

    # Sort by the magnitude of importance and select the top k
    sorted_by_importance = sorted(feature_importance.items(),
                                  key=lambda x: abs(x[1]),
                                  reverse=True)
    top_k_feats = dict(sorted_by_importance[:k])
    
    logger.info("Returning top %d features out of %d total.", k, len(feature_importance))
    return top_k_feats

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

def get_score(y_true, y_pred_proba, trained_classes=None):
    # If 2D array of probabilities is provided, get probabilities for positive class
    if y_pred_proba.ndim == 2:
        if trained_classes is not None:
            # Get index of positive class (1)
            pos_idx = np.where(trained_classes == 1)[0][0]
            y_pred_proba = y_pred_proba[:, pos_idx]
        else:
            # Default to second column for positive class
            y_pred_proba = y_pred_proba[:, 1]
            
    return roc_auc_score(y_true, y_pred_proba)

def get_data(split, schema=None):
    """
    Retrieve the microbiome dataset for a given split (train or validate).
    Returns:
        X (pd.DataFrame): Feature matrix with taxa readcounts.
        y (pd.Series): Binary labels (0 = Healthy, 1 = UC/CD) for each sample.
    """
    # Connect to database
    engine = sqlite3.connect(DB_PATH)
    
    # Assume `config` is a global dict from the YAML configuration and `engine` is a database connection.
    cohorts = CONFIG['split_cohorts'][split]  # e.g., list of cohort names for the split
    # Format cohort list for SQL IN clause
    cohort_list_str = "', '".join(cohorts)
    
    # SQL query to get all nonzero read counts for the selected runs with taxon names
    query_seq = f""" 
        SELECT g.run, t.taxon_name AS taxon, g.rpm
        FROM genomic_sequence_rpm AS g
        JOIN selected_runs AS s
          ON g.run = s.run
        JOIN taxa t
          ON g.taxon_id = t.taxon_id
        WHERE s.cohort IN ('{cohort_list_str}')
    """
    
    # Read data into pandas DataFrame
    df = pd.read_sql(query_seq, engine)
    
    # Pivot table to get features (taxa) as columns
    X = df.pivot(index='run', columns='taxon', values='rpm').fillna(0)
    
    # Get labels for the runs
    query_labels = f"""
        SELECT run, 
               CASE WHEN condition = 'Healthy' THEN 0 ELSE 1 END as label
        FROM selected_runs
        WHERE cohort IN ('{cohort_list_str}')
    """
    y = pd.read_sql(query_labels, engine).set_index('run')['label']
    
    # Ensure X and y have same indices in same order
    y = y.reindex(X.index)
    
    engine.close()
    return X, y
    
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