"""
Serialization utilities for the Autonomous Trading System.

This module provides utilities for serializing and deserializing data,
including JSON, pickle, and other formats.
"""

import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("utils.serialization.serialization_utils")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        return super().default(obj)


def save_json(
    data: Any,
    file_path: str,
    indent: Optional[int] = None,
    ensure_dir: bool = True,
) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the data to
        indent: Indentation level for pretty printing
        ensure_dir: Whether to ensure the directory exists
    """
    try:
        if ensure_dir:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=indent)
        
        logger.debug(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise


def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to load the data from
        
    Returns:
        Loaded data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.debug(f"Data loaded from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def save_pickle(
    data: Any,
    file_path: str,
    protocol: int = pickle.HIGHEST_PROTOCOL,
    ensure_dir: bool = True,
) -> None:
    """
    Save data to a pickle file.
    
    Args:
        data: Data to save
        file_path: Path to save the data to
        protocol: Pickle protocol version
        ensure_dir: Whether to ensure the directory exists
    """
    try:
        if ensure_dir:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=protocol)
        
        logger.debug(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise


def load_pickle(file_path: str) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        file_path: Path to load the data from
        
    Returns:
        Loaded data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.debug(f"Data loaded from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def save_dataframe(
    df: pd.DataFrame,
    file_path: str,
    format: str = 'csv',
    ensure_dir: bool = True,
    **kwargs,
) -> None:
    """
    Save a DataFrame to a file.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the DataFrame to
        format: File format ('csv', 'parquet', 'hdf', 'json', 'pickle')
        ensure_dir: Whether to ensure the directory exists
        **kwargs: Additional arguments to pass to the save function
    """
    try:
        if ensure_dir:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        format = format.lower()
        
        if format == 'csv':
            df.to_csv(file_path, **kwargs)
        elif format == 'parquet':
            df.to_parquet(file_path, **kwargs)
        elif format == 'hdf':
            df.to_hdf(file_path, key='data', **kwargs)
        elif format == 'json':
            df.to_json(file_path, **kwargs)
        elif format == 'pickle':
            df.to_pickle(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug(f"DataFrame saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {file_path}: {e}")
        raise


def load_dataframe(
    file_path: str,
    format: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load a DataFrame from a file.
    
    Args:
        file_path: Path to load the DataFrame from
        format: File format ('csv', 'parquet', 'hdf', 'json', 'pickle')
        **kwargs: Additional arguments to pass to the load function
        
    Returns:
        Loaded DataFrame
    """
    try:
        # Determine format from file extension if not provided
        if format is None:
            _, ext = os.path.splitext(file_path)
            format = ext.lstrip('.').lower()
        else:
            format = format.lower()
        
        if format == 'csv':
            df = pd.read_csv(file_path, **kwargs)
        elif format in ('parquet', 'pq'):
            df = pd.read_parquet(file_path, **kwargs)
        elif format in ('hdf', 'h5'):
            df = pd.read_hdf(file_path, **kwargs)
        elif format == 'json':
            df = pd.read_json(file_path, **kwargs)
        elif format in ('pickle', 'pkl'):
            df = pd.read_pickle(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug(f"DataFrame loaded from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from {file_path}: {e}")
        raise


def dataframe_to_dict(
    df: pd.DataFrame,
    orient: str = 'records',
    date_format: str = 'iso',
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convert a DataFrame to a dictionary.
    
    Args:
        df: DataFrame to convert
        orient: Orientation of the result ('records', 'list', 'dict', 'series', 'split', 'index')
        date_format: Date format ('iso', 'epoch')
        
    Returns:
        Dictionary representation of the DataFrame
    """
    try:
        return df.to_dict(orient=orient, date_format=date_format)
    except Exception as e:
        logger.error(f"Error converting DataFrame to dictionary: {e}")
        raise


def dict_to_dataframe(
    data: Union[List[Dict[str, Any]], Dict[str, Any]],
    orient: str = 'records',
) -> pd.DataFrame:
    """
    Convert a dictionary to a DataFrame.
    
    Args:
        data: Dictionary to convert
        orient: Orientation of the input ('records', 'list', 'dict', 'series', 'split', 'index')
        
    Returns:
        DataFrame representation of the dictionary
    """
    try:
        return pd.DataFrame.from_dict(data, orient=orient)
    except Exception as e:
        logger.error(f"Error converting dictionary to DataFrame: {e}")
        raise


def serialize_numpy(
    array: np.ndarray,
    file_path: str,
    format: str = 'npy',
    ensure_dir: bool = True,
    **kwargs,
) -> None:
    """
    Serialize a NumPy array to a file.
    
    Args:
        array: NumPy array to serialize
        file_path: Path to save the array to
        format: File format ('npy', 'npz', 'txt')
        ensure_dir: Whether to ensure the directory exists
        **kwargs: Additional arguments to pass to the save function
    """
    try:
        if ensure_dir:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        format = format.lower()
        
        if format == 'npy':
            np.save(file_path, array, **kwargs)
        elif format == 'npz':
            np.savez(file_path, array, **kwargs)
        elif format == 'txt':
            np.savetxt(file_path, array, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug(f"NumPy array saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving NumPy array to {file_path}: {e}")
        raise


def deserialize_numpy(
    file_path: str,
    format: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """
    Deserialize a NumPy array from a file.
    
    Args:
        file_path: Path to load the array from
        format: File format ('npy', 'npz', 'txt')
        **kwargs: Additional arguments to pass to the load function
        
    Returns:
        Loaded NumPy array
    """
    try:
        # Determine format from file extension if not provided
        if format is None:
            _, ext = os.path.splitext(file_path)
            format = ext.lstrip('.').lower()
        else:
            format = format.lower()
        
        if format == 'npy':
            array = np.load(file_path, **kwargs)
        elif format == 'npz':
            with np.load(file_path, **kwargs) as data:
                # Get the first array from the archive
                array = data[list(data.keys())[0]]
        elif format == 'txt':
            array = np.loadtxt(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug(f"NumPy array loaded from {file_path}")
        return array
    except Exception as e:
        logger.error(f"Error loading NumPy array from {file_path}: {e}")
        raise


def serialize_model(
    model: Any,
    file_path: str,
    ensure_dir: bool = True,
) -> None:
    """
    Serialize a model to a file.
    
    Args:
        model: Model to serialize
        file_path: Path to save the model to
        ensure_dir: Whether to ensure the directory exists
    """
    try:
        if ensure_dir:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Check if the model has a save method
        if hasattr(model, 'save') and callable(getattr(model, 'save')):
            model.save(file_path)
        else:
            # Fall back to pickle
            save_pickle(model, file_path)
        
        logger.debug(f"Model saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise


def deserialize_model(
    file_path: str,
    custom_objects: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Deserialize a model from a file.
    
    Args:
        file_path: Path to load the model from
        custom_objects: Dictionary mapping names to custom classes or functions
        
    Returns:
        Loaded model
    """
    try:
        # Try to determine the model type from the file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lstrip('.').lower()
        
        # Check if it's a TensorFlow/Keras model
        if ext in ('h5', 'keras', 'tf'):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(file_path, custom_objects=custom_objects)
                logger.debug(f"TensorFlow model loaded from {file_path}")
                return model
            except ImportError:
                logger.warning("TensorFlow not installed, falling back to pickle")
            except Exception as e:
                logger.warning(f"Error loading as TensorFlow model: {e}, falling back to pickle")
        
        # Check if it's an XGBoost model
        if ext in ('xgb', 'bst'):
            try:
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(file_path)
                logger.debug(f"XGBoost model loaded from {file_path}")
                return model
            except ImportError:
                logger.warning("XGBoost not installed, falling back to pickle")
            except Exception as e:
                logger.warning(f"Error loading as XGBoost model: {e}, falling back to pickle")
        
        # Check if it's a scikit-learn model
        if ext in ('joblib', 'pkl'):
            try:
                from joblib import load
                model = load(file_path)
                logger.debug(f"scikit-learn model loaded from {file_path}")
                return model
            except ImportError:
                logger.warning("joblib not installed, falling back to pickle")
            except Exception as e:
                logger.warning(f"Error loading as scikit-learn model: {e}, falling back to pickle")
        
        # Fall back to pickle
        model = load_pickle(file_path)
        logger.debug(f"Model loaded from {file_path} using pickle")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        raise


def object_to_json_string(
    obj: Any,
    indent: Optional[int] = None,
) -> str:
    """
    Convert an object to a JSON string.
    
    Args:
        obj: Object to convert
        indent: Indentation level for pretty printing
        
    Returns:
        JSON string representation of the object
    """
    try:
        return json.dumps(obj, cls=NumpyEncoder, indent=indent)
    except Exception as e:
        logger.error(f"Error converting object to JSON string: {e}")
        raise


def json_string_to_object(json_str: str) -> Any:
    """
    Convert a JSON string to an object.
    
    Args:
        json_str: JSON string to convert
        
    Returns:
        Object representation of the JSON string
    """
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Error converting JSON string to object: {e}")
        raise