"""
Data Transformer Module

This module provides transformation functions for data received from various sources
(Polygon, Unusual Whales, etc.) before it's stored in the database. It ensures proper
data typing, validation, and consistency.
"""

import json
import numpy as np
import pandas as pd

# Import logging utility
from src.utils.logging import get_logger

# Configure logging
logger = get_logger("data_acquisition.transformation.data_transformer")


class DataTransformer:
    """
    Transforms data from various sources to ensure it matches the database schema
    before storage. This eliminates the need for temporary tables and improves
    data consistency.
    """

    @staticmethod
    def transform_stock_aggs(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform stock aggregates data.
        
        Args:
            df: DataFrame with stock aggregates data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["timestamp"]):
            transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"])
            
        # Convert string columns
        string_cols = ["symbol", "timeframe", "source", "timespan_unit"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "vwap"]
        for col in numeric_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce")
                
        # Convert integer columns
        int_cols = ["volume", "transactions", "multiplier"]
        for col in int_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce").fillna(0).astype(np.int64)
                
        # Convert boolean columns
        bool_cols = ["adjusted", "otc"]
        for col in bool_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(bool)
                
        return transformed_df
        
    @staticmethod
    def transform_crypto_aggs(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform crypto aggregates data.
        
        Args:
            df: DataFrame with crypto aggregates data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["timestamp"]):
            transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"])
            
        # Convert string columns
        string_cols = ["symbol", "timeframe", "source", "timespan_unit", "exchange"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert numeric columns (crypto uses higher precision)
        numeric_cols = ["open", "high", "low", "close", "volume", "vwap"]
        for col in numeric_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce")
                
        # Convert integer columns
        int_cols = ["transactions", "multiplier"]
        for col in int_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce").fillna(0).astype(np.int64)
                
        return transformed_df
        
    @staticmethod
    def transform_quotes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform quotes data.
        
        Args:
            df: DataFrame with quotes data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["timestamp"]):
            transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"])
            
        # Convert string columns
        string_cols = ["symbol", "exchange", "source"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert numeric columns
        numeric_cols = ["bid_price", "ask_price"]
        for col in numeric_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce")
                
        # Convert integer columns
        int_cols = ["bid_size", "ask_size", "sequence_number"]
        for col in int_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce").fillna(0).astype(np.int64)
                
        # Handle tape column (should be a single character)
        if "tape" in transformed_df.columns:
            transformed_df["tape"] = transformed_df["tape"].astype(str).str[:1]
            
        # Handle conditions array
        if "conditions" in transformed_df.columns and not isinstance(transformed_df["conditions"].iloc[0], list):
            transformed_df["conditions"] = transformed_df["conditions"].apply(
                lambda x: [] if pd.isna(x) else (json.loads(x) if isinstance(x, str) else x)
            )
            
        return transformed_df
        
    @staticmethod
    def transform_trades(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform trades data.
        
        Args:
            df: DataFrame with trades data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["timestamp"]):
            transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"])
            
        # Convert string columns
        string_cols = ["symbol", "exchange", "trade_id", "source"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert numeric columns
        if "price" in transformed_df.columns:
            transformed_df["price"] = pd.to_numeric(transformed_df["price"], errors="coerce")
                
        # Convert integer columns
        int_cols = ["size", "sequence_number"]
        for col in int_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce").fillna(0).astype(np.int64)
                
        # Handle tape column (should be a single character)
        if "tape" in transformed_df.columns:
            transformed_df["tape"] = transformed_df["tape"].astype(str).str[:1]
            
        # Handle conditions array
        if "conditions" in transformed_df.columns and not isinstance(transformed_df["conditions"].iloc[0], list):
            transformed_df["conditions"] = transformed_df["conditions"].apply(
                lambda x: [] if pd.isna(x) else (json.loads(x) if isinstance(x, str) else x)
            )
            
        return transformed_df
        
    @staticmethod
    def transform_options_aggs(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform options aggregates data.
        
        Args:
            df: DataFrame with options aggregates data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["timestamp"]):
            transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"])
            
        # Ensure expiration is in date format
        if "expiration" in transformed_df.columns and not pd.api.types.is_datetime64_any_dtype(transformed_df["expiration"]):
            transformed_df["expiration"] = pd.to_datetime(transformed_df["expiration"]).dt.date
            
        # Convert string columns
        string_cols = ["symbol", "underlying", "timeframe", "source", "timespan_unit"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert option_type to single character (C for call, P for put)
        if "option_type" in transformed_df.columns:
            transformed_df["option_type"] = transformed_df["option_type"].apply(
                lambda x: "C" if str(x).lower() == "call" else "P" if str(x).lower() == "put" else str(x)[:1]
            )
                
        # Convert numeric columns
        numeric_cols = ["strike", "open", "high", "low", "close"]
        for col in numeric_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce")
                
        # Convert integer columns
        int_cols = ["volume", "open_interest", "multiplier"]
        for col in int_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce").fillna(0).astype(np.int64)
                
        return transformed_df
        
    @staticmethod
    def transform_options_flow(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform options flow data from Unusual Whales.
        
        Args:
            df: DataFrame with options flow data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["timestamp"]):
            transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"])
            
        # Ensure expiration_date is in datetime format if present
        if "expiration_date" in transformed_df.columns and not pd.api.types.is_datetime64_any_dtype(transformed_df["expiration_date"]):
            transformed_df["expiration_date"] = pd.to_datetime(transformed_df["expiration_date"])
            
        # Convert string columns
        string_cols = ["id", "symbol", "contract_type", "sentiment", "trade_type", "source"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert numeric columns
        numeric_cols = ["strike", "premium", "implied_volatility", "delta", "gamma", "theta", "vega"]
        for col in numeric_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce")
                
        # Convert integer columns
        int_cols = ["size", "open_interest"]
        for col in int_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce").fillna(0).astype(np.int64)
                
        return transformed_df
        
    @staticmethod
    def transform_ticker_details(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform ticker details data.
        
        Args:
            df: DataFrame with ticker details
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure last_updated is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["last_updated"]):
            transformed_df["last_updated"] = pd.to_datetime(transformed_df["last_updated"])
            
        # Convert string columns
        string_cols = [
            "ticker", "name", "market", "locale", "type", "currency",
            "primary_exchange", "description", "sic_code", "sic_description",
            "ticker_root", "homepage_url", "phone_number"
        ]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert boolean columns
        if "active" in transformed_df.columns:
            transformed_df["active"] = transformed_df["active"].astype(bool)
            
        # Convert date columns
        if "list_date" in transformed_df.columns:
            transformed_df["list_date"] = pd.to_datetime(transformed_df["list_date"], errors="coerce")
            
        # Convert numeric columns with proper error handling
        if "total_employees" in transformed_df.columns:
            transformed_df["total_employees"] = pd.to_numeric(transformed_df["total_employees"], errors="coerce").fillna(0).astype(np.int64)
            
        # Convert big integer columns
        big_int_cols = ["share_class_shares_outstanding", "weighted_shares_outstanding", "market_cap"]
        for col in big_int_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce").fillna(0).astype(np.int64)
                
        # Handle JSON columns
        for json_col in ["address", "metadata"]:
            if json_col in transformed_df.columns:
                # Convert to string representation if not already
                transformed_df[json_col] = transformed_df[json_col].apply(
                    lambda x: "{}" if pd.isna(x) or x == "" else (json.dumps(x) if not isinstance(x, str) else x)
                )
                
        return transformed_df
        
    @staticmethod
    def transform_market_status(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform market status data.
        
        Args:
            df: DataFrame with market status data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["timestamp"]):
            transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"])
            
        # Ensure next_open and next_close are in datetime format
        for col in ["next_open", "next_close"]:
            if col in transformed_df.columns and not pd.api.types.is_datetime64_any_dtype(transformed_df[col]):
                transformed_df[col] = pd.to_datetime(transformed_df[col], errors="coerce")
                
        # Convert string columns
        string_cols = ["market", "status"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert boolean columns
        bool_cols = ["early_close", "late_open"]
        for col in bool_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(bool)
                
        return transformed_df
        
    @staticmethod
    def transform_market_holidays(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform market holidays data.
        
        Args:
            df: DataFrame with market holidays data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure date is in date format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["date"]):
            transformed_df["date"] = pd.to_datetime(transformed_df["date"]).dt.date
            
        # Convert time columns if present
        for col in ["open_time", "close_time"]:
            if col in transformed_df.columns and not pd.api.types.is_datetime64_any_dtype(transformed_df[col]):
                transformed_df[col] = pd.to_datetime(transformed_df[col], errors="coerce")
                
        # Convert string columns
        string_cols = ["name", "market", "status"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert year to integer
        if "year" in transformed_df.columns:
            transformed_df["year"] = pd.to_numeric(transformed_df["year"], errors="coerce").fillna(0).astype(np.int64)
                
        return transformed_df
        
    @staticmethod
    def transform_news_articles(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform news articles data.
        
        Args:
            df: DataFrame with news articles data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure published_utc is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["published_utc"]):
            transformed_df["published_utc"] = pd.to_datetime(transformed_df["published_utc"])
            
        # Convert string columns
        string_cols = ["article_id", "title", "author", "article_url", "image_url", "description", "source"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Handle array columns
        for array_col in ["tickers", "keywords"]:
            if array_col in transformed_df.columns and not isinstance(transformed_df[array_col].iloc[0], list):
                transformed_df[array_col] = transformed_df[array_col].apply(
                    lambda x: [] if pd.isna(x) else (json.loads(x) if isinstance(x, str) else x)
                )
                
        return transformed_df
        
    @staticmethod
    def transform_news_sentiment(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform news sentiment data.
        
        Args:
            df: DataFrame with news sentiment data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["timestamp"]):
            transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"])
            
        # Convert string columns
        string_cols = ["article_id", "symbol", "sentiment_label", "model_version"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert numeric columns
        numeric_cols = ["sentiment_score", "confidence"]
        for col in numeric_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce")
                
        # Handle JSON columns
        for json_col in ["entity_mentions", "keywords"]:
            if json_col in transformed_df.columns:
                transformed_df[json_col] = transformed_df[json_col].apply(
                    lambda x: [] if pd.isna(x) else (json.loads(x) if isinstance(x, str) else x)
                )
                
        return transformed_df
        
    @staticmethod
    def transform_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform feature data.
        
        Args:
            df: DataFrame with feature data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["timestamp"]):
            transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"])
            
        # Convert string columns
        string_cols = ["symbol", "feature_name", "timeframe", "feature_group"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert numeric columns
        if "feature_value" in transformed_df.columns:
            transformed_df["feature_value"] = pd.to_numeric(transformed_df["feature_value"], errors="coerce")
                
        return transformed_df
        
    @staticmethod
    def transform_feature_metadata(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform feature metadata.
        
        Args:
            df: DataFrame with feature metadata
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamps are in datetime format
        for col in ["created_at", "updated_at"]:
            if col in transformed_df.columns and not pd.api.types.is_datetime64_any_dtype(transformed_df[col]):
                transformed_df[col] = pd.to_datetime(transformed_df[col])
                
        # Convert string columns
        string_cols = ["feature_name", "description", "formula", "version"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert boolean columns
        if "is_active" in transformed_df.columns:
            transformed_df["is_active"] = transformed_df["is_active"].astype(bool)
            
        # Handle JSON columns
        if "parameters" in transformed_df.columns:
            transformed_df["parameters"] = transformed_df["parameters"].apply(
                lambda x: {} if pd.isna(x) else (json.loads(x) if isinstance(x, str) else x)
            )
                
        return transformed_df
        
    @staticmethod
    def transform_models(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform model data.
        
        Args:
            df: DataFrame with model data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamps are in datetime format
        for col in ["created_at", "trained_at"]:
            if col in transformed_df.columns and not pd.api.types.is_datetime64_any_dtype(transformed_df[col]):
                transformed_df[col] = pd.to_datetime(transformed_df[col])
                
        # Convert string columns
        string_cols = ["model_id", "model_name", "model_type", "target", "version", "status", "file_path"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Handle array columns
        if "features" in transformed_df.columns and not isinstance(transformed_df["features"].iloc[0], list):
            transformed_df["features"] = transformed_df["features"].apply(
                lambda x: [] if pd.isna(x) else (json.loads(x) if isinstance(x, str) else x)
            )
            
        # Handle JSON columns
        for json_col in ["parameters", "metrics"]:
            if json_col in transformed_df.columns:
                transformed_df[json_col] = transformed_df[json_col].apply(
                    lambda x: {} if pd.isna(x) else (json.loads(x) if isinstance(x, str) else x)
                )
                
        return transformed_df
        
    @staticmethod
    def transform_model_training_runs(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform model training run data.
        
        Args:
            df: DataFrame with model training run data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamps are in datetime format
        for col in ["start_time", "end_time"]:
            if col in transformed_df.columns and not pd.api.types.is_datetime64_any_dtype(transformed_df[col]):
                transformed_df[col] = pd.to_datetime(transformed_df[col])
                
        # Convert string columns
        string_cols = ["run_id", "model_id", "status", "logs"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Handle JSON columns
        for json_col in ["parameters", "metrics"]:
            if json_col in transformed_df.columns:
                transformed_df[json_col] = transformed_df[json_col].apply(
                    lambda x: {} if pd.isna(x) else (json.loads(x) if isinstance(x, str) else x)
                )
                
        return transformed_df
        
    @staticmethod
    def transform_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform trading signal data.
        
        Args:
            df: DataFrame with trading signal data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["timestamp"]):
            transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"])
            
        # Convert string columns
        string_cols = ["signal_id", "symbol", "signal_type", "model_id", "timeframe"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert numeric columns
        if "confidence" in transformed_df.columns:
            transformed_df["confidence"] = pd.to_numeric(transformed_df["confidence"], errors="coerce")
            
        # Handle JSON columns
        for json_col in ["parameters", "features_snapshot"]:
            if json_col in transformed_df.columns:
                transformed_df[json_col] = transformed_df[json_col].apply(
                    lambda x: {} if pd.isna(x) else (json.loads(x) if isinstance(x, str) else x)
                )
                
        return transformed_df
        
    @staticmethod
    def transform_orders(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform order data.
        
        Args:
            df: DataFrame with order data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamps are in datetime format
        for col in ["timestamp", "updated_at"]:
            if col in transformed_df.columns and not pd.api.types.is_datetime64_any_dtype(transformed_df[col]):
                transformed_df[col] = pd.to_datetime(transformed_df[col])
                
        # Convert string columns
        string_cols = ["order_id", "external_order_id", "symbol", "order_type", "side", "signal_id", "strategy_id", "status"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert numeric columns
        numeric_cols = ["quantity", "price", "filled_quantity", "filled_price", "commission"]
        for col in numeric_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce")
                
        return transformed_df
        
    @staticmethod
    def transform_positions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform position data.
        
        Args:
            df: DataFrame with position data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamps are in datetime format
        for col in ["entry_time", "last_update"]:
            if col in transformed_df.columns and not pd.api.types.is_datetime64_any_dtype(transformed_df[col]):
                transformed_df[col] = pd.to_datetime(transformed_df[col])
                
        # Convert string columns
        string_cols = ["position_id", "symbol", "strategy_id", "status"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert numeric columns
        numeric_cols = ["quantity", "entry_price", "current_price", "pnl", "pnl_percentage"]
        for col in numeric_cols:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce")
                
        # Handle JSON columns
        if "metadata" in transformed_df.columns:
            transformed_df["metadata"] = transformed_df["metadata"].apply(
                lambda x: {} if pd.isna(x) else (json.loads(x) if isinstance(x, str) else x)
            )
                
        return transformed_df
        
    @staticmethod
    def transform_system_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform system metrics data.
        
        Args:
            df: DataFrame with system metrics data
            
        Returns:
            Transformed DataFrame ready for database storage
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transformed_df["timestamp"]):
            transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"])
            
        # Convert string columns
        string_cols = ["metric_name", "component", "host"]
        for col in string_cols:
            if col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].astype(str)
                
        # Convert numeric columns
        if "metric_value" in transformed_df.columns:
            transformed_df["metric_value"] = pd.to_numeric(transformed_df["metric_value"], errors="coerce")
            
        # Handle JSON columns
        if "tags" in transformed_df.columns:
            transformed_df["tags"] = transformed_df["tags"].apply(
                lambda x: {} if pd.isna(x) else (json.loads(x) if isinstance(x, str) else x)
            )
                
        return transformed_df