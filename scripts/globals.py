# globals.py
import os
import re
import sys
import copy
import time
import yaml
import signal
import logging
import traceback
import gc
import hashlib
import random
import pickle
import shutil
import argparse
import zipfile
import math
import sqlite3
from tqdm import tqdm
from typing import List, Any, Optional
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple
from itertools import product, zip_longest
from pathlib import Path
from datetime import datetime
from importlib import import_module

from filelock import FileLock

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Setup directories and paths as constants
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
CONFIGS_DIR = BASE_DIR / 'configs'
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / 'data'
SCRIPTS_DIR = BASE_DIR / 'scripts'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
SCORES_DIR = RESULTS_DIR / 'scored_model_configs'
WORKING_DATA_DIR = DATA_DIR / 'working'
SCORES_SAVE_QUEUE_PATH = RESULTS_DIR / 'scores_queue'

# Load configuration as a constant
with open(CONFIGS_DIR / 'config.yaml', 'r') as f:
    CONFIG =  yaml.safe_load(f)
PIPE_CONFIG_PATH = BASE_DIR / CONFIG['pipe_config_path']
SEED = CONFIG['seed']
N_SPLITS = CONFIG['n_splits']
N_EXTRACT_FEATURES = CONFIG['n_extract_features']
LOG_NAME = CONFIG['log_name']
DB_PATH = DATA_DIR / CONFIG['db_name']