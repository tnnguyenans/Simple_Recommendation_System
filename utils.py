#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the recommendation system.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vector1, vector2) / (norm1 * norm2)


def pearson_correlation(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient between two vectors.
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Pearson correlation coefficient between -1 and 1
    """
    if len(vector1) < 2 or len(vector2) < 2:
        return 0.0
    
    # Remove pairs where either value is 0
    mask = ~((vector1 == 0) | (vector2 == 0))
    filtered1 = vector1[mask]
    filtered2 = vector2[mask]
    
    if len(filtered1) < 2:
        return 0.0
    
    return np.corrcoef(filtered1, filtered2)[0, 1]


def evaluate_recommendations(
    predicted_ratings: Dict[int, float], 
    actual_ratings: Dict[int, float]
) -> Dict[str, float]:
    """
    Evaluate recommendation quality using multiple metrics.
    
    Args:
        predicted_ratings: Dictionary of item_id to predicted rating
        actual_ratings: Dictionary of item_id to actual rating
        
    Returns:
        Dictionary with evaluation metrics (RMSE, MAE, precision, recall)
    """
    # Get common item IDs
    common_items = set(predicted_ratings.keys()) & set(actual_ratings.keys())
    
    if not common_items:
        logger.warning("No common items between predicted and actual ratings")
        return {
            "rmse": 0.0,
            "mae": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }
    
    # Calculate RMSE and MAE
    squared_errors = 0.0
    absolute_errors = 0.0
    
    for item_id in common_items:
        pred = predicted_ratings[item_id]
        actual = actual_ratings[item_id]
        
        squared_errors += (pred - actual) ** 2
        absolute_errors += abs(pred - actual)
    
    rmse = np.sqrt(squared_errors / len(common_items))
    mae = absolute_errors / len(common_items)
    
    # Calculate precision and recall (considering items with rating >= 4 as relevant)
    relevant_threshold = 4.0
    
    actual_relevant = {k for k, v in actual_ratings.items() if v >= relevant_threshold}
    predicted_relevant = {k for k, v in predicted_ratings.items() if v >= relevant_threshold}
    
    if not predicted_relevant:
        precision = 0.0
    else:
        precision = len(actual_relevant & predicted_relevant) / len(predicted_relevant)
    
    if not actual_relevant:
        recall = 0.0
    else:
        recall = len(actual_relevant & predicted_relevant) / len(actual_relevant)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "precision": precision,
        "recall": recall
    }


def normalize_ratings(ratings: np.ndarray) -> np.ndarray:
    """
    Normalize ratings by subtracting the mean of each user's ratings.
    
    Args:
        ratings: User-item ratings matrix
        
    Returns:
        Normalized ratings matrix
    """
    # Calculate mean rating for each user (row)
    user_means = np.true_divide(
        np.sum(ratings, axis=1),
        np.maximum(np.count_nonzero(ratings, axis=1), 1)
    ).reshape(-1, 1)
    
    # Create a copy to avoid modifying the original
    normalized = ratings.copy()
    
    # Only normalize non-zero entries
    mask = ratings != 0
    normalized[mask] = ratings[mask] - np.repeat(user_means, ratings.shape[1], axis=1)[mask]
    
    return normalized
