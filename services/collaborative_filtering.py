#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collaborative filtering recommendation algorithm implementation.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from utils import cosine_similarity, pearson_correlation


class CollaborativeFiltering:
    """
    Collaborative filtering recommendation algorithm using user-based and item-based approaches.
    """
    
    def __init__(self, method: str = 'user-based', similarity_metric: str = 'cosine'):
        """
        Initialize the collaborative filtering algorithm.
        
        Args:
            method: Method to use ('user-based' or 'item-based')
            similarity_metric: Similarity metric ('cosine' or 'pearson')
        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.info(f"Initializing {method} collaborative filtering with {similarity_metric} similarity")
        
        self.method = method
        self.similarity_metric = similarity_metric
        
        # Data storage
        self.user_item_ratings = {}  # {user_id: {item_id: rating}}
        self.item_user_ratings = {}  # {item_id: {user_id: rating}}
        self.user_means = {}         # {user_id: mean_rating}
        self.item_means = {}         # {item_id: mean_rating}
        
        # Similarity matrices
        self.user_similarity = {}    # {user_id: {other_user_id: similarity}}
        self.item_similarity = {}    # {item_id: {other_item_id: similarity}}
        
        # Choose similarity function
        self.similarity_func = cosine_similarity if similarity_metric == 'cosine' else pearson_correlation
    
    def train(self, data: Dict[str, Any]) -> None:
        """
        Train the algorithm with rating data.
        
        Args:
            data: Dictionary with 'ratings' list of Rating objects
        """
        self._logger.info("Training collaborative filtering model")
        
        # Reset data structures
        self.user_item_ratings = {}
        self.item_user_ratings = {}
        self.user_means = {}
        self.item_means = {}
        
        # Process ratings
        ratings = data.get('ratings', [])
        self._logger.debug(f"Processing {len(ratings)} ratings")
        
        for rating in ratings:
            user_id = rating.user_id
            item_id = rating.item_id
            value = float(rating.value)
            
            # Add to user-item matrix
            if user_id not in self.user_item_ratings:
                self.user_item_ratings[user_id] = {}
            self.user_item_ratings[user_id][item_id] = value
            
            # Add to item-user matrix
            if item_id not in self.item_user_ratings:
                self.item_user_ratings[item_id] = {}
            self.item_user_ratings[item_id][user_id] = value
        
        # Calculate user means
        for user_id, ratings in self.user_item_ratings.items():
            if ratings:
                self.user_means[user_id] = sum(ratings.values()) / len(ratings)
            else:
                self.user_means[user_id] = 0.0
        
        # Calculate item means
        for item_id, ratings in self.item_user_ratings.items():
            if ratings:
                self.item_means[item_id] = sum(ratings.values()) / len(ratings)
            else:
                self.item_means[item_id] = 0.0
        
        # Calculate similarities based on chosen method
        if self.method == 'user-based':
            self._compute_user_similarities()
        else:  # item-based
            self._compute_item_similarities()
            
        self._logger.info("Collaborative filtering model training complete")
    
    def _compute_user_similarities(self) -> None:
        """Compute similarity between all users."""
        self._logger.debug("Computing user similarities")
        self.user_similarity = {}
        
        all_users = list(self.user_item_ratings.keys())
        all_items = set()
        for user_ratings in self.user_item_ratings.values():
            all_items.update(user_ratings.keys())
        all_items = list(all_items)
        
        # Create user vectors for all items
        user_vectors = {}
        for user_id in all_users:
            vector = []
            user_ratings = self.user_item_ratings.get(user_id, {})
            for item_id in all_items:
                vector.append(user_ratings.get(item_id, 0.0))
            user_vectors[user_id] = np.array(vector)
        
        # Calculate similarities
        for i, user1 in enumerate(all_users):
            self.user_similarity[user1] = {}
            vec1 = user_vectors[user1]
            
            for user2 in all_users:
                if user1 == user2:
                    continue
                    
                vec2 = user_vectors[user2]
                similarity = self.similarity_func(vec1, vec2)
                
                # Only store significant similarities to save memory
                if not np.isnan(similarity) and similarity > 0.1:
                    self.user_similarity[user1][user2] = similarity
    
    def _compute_item_similarities(self) -> None:
        """Compute similarity between all items."""
        self._logger.debug("Computing item similarities")
        self.item_similarity = {}
        
        all_items = list(self.item_user_ratings.keys())
        all_users = set()
        for item_ratings in self.item_user_ratings.values():
            all_users.update(item_ratings.keys())
        all_users = list(all_users)
        
        # Create item vectors for all users
        item_vectors = {}
        for item_id in all_items:
            vector = []
            item_ratings = self.item_user_ratings.get(item_id, {})
            for user_id in all_users:
                vector.append(item_ratings.get(user_id, 0.0))
            item_vectors[item_id] = np.array(vector)
        
        # Calculate similarities
        for i, item1 in enumerate(all_items):
            self.item_similarity[item1] = {}
            vec1 = item_vectors[item1]
            
            for item2 in all_items:
                if item1 == item2:
                    continue
                    
                vec2 = item_vectors[item2]
                similarity = self.similarity_func(vec1, vec2)
                
                # Only store significant similarities to save memory
                if not np.isnan(similarity) and similarity > 0.1:
                    self.item_similarity[item1][item2] = similarity
    
    def recommend_for_user(self, user_id: int, limit: int = 10) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID to recommend for
            limit: Maximum number of recommendations
            
        Returns:
            List of tuples (item_id, predicted_rating)
        """
        self._logger.info(f"Generating recommendations for user {user_id}")
        
        if self.method == 'user-based':
            return self._user_based_recommendations(user_id, limit)
        else:
            return self._item_based_recommendations(user_id, limit)
    
    def _user_based_recommendations(self, user_id: int, limit: int = 10) -> List[Tuple[int, float]]:
        """Generate user-based recommendations."""
        # Get user's ratings
        user_ratings = self.user_item_ratings.get(user_id, {})
        if not user_ratings:
            self._logger.warning(f"No ratings found for user {user_id}")
            return []
        
        # User's rated items
        rated_items = set(user_ratings.keys())
        
        # Get similar users
        similar_users = self.user_similarity.get(user_id, {})
        if not similar_users:
            self._logger.warning(f"No similar users found for user {user_id}")
            return []
        
        # Generate predictions for unrated items
        predictions = {}
        user_mean = self.user_means.get(user_id, 0.0)
        
        for other_user_id, similarity in sorted(similar_users.items(), key=lambda x: x[1], reverse=True):
            other_user_ratings = self.user_item_ratings.get(other_user_id, {})
            other_user_mean = self.user_means.get(other_user_id, 0.0)
            
            for item_id, rating in other_user_ratings.items():
                if item_id in rated_items:
                    continue  # Skip items the user has already rated
                
                if item_id not in predictions:
                    predictions[item_id] = {'weighted_sum': 0.0, 'similarity_sum': 0.0}
                
                # Normalize rating by user's mean
                adjusted_rating = rating - other_user_mean
                
                # Update prediction
                predictions[item_id]['weighted_sum'] += similarity * adjusted_rating
                predictions[item_id]['similarity_sum'] += abs(similarity)
        
        # Calculate final predictions
        final_predictions = []
        for item_id, data in predictions.items():
            if data['similarity_sum'] > 0:
                # Denormalize prediction using the target user's mean
                prediction = user_mean + (data['weighted_sum'] / data['similarity_sum'])
                
                # Clamp to valid rating range
                prediction = max(0.5, min(5.0, prediction))
                final_predictions.append((item_id, prediction))
        
        # Sort and return top recommendations
        final_predictions.sort(key=lambda x: x[1], reverse=True)
        return final_predictions[:limit]
    
    def _item_based_recommendations(self, user_id: int, limit: int = 10) -> List[Tuple[int, float]]:
        """Generate item-based recommendations."""
        # Get user's ratings
        user_ratings = self.user_item_ratings.get(user_id, {})
        if not user_ratings:
            self._logger.warning(f"No ratings found for user {user_id}")
            return []
        
        # User's rated items
        rated_items = set(user_ratings.keys())
        all_items = set(self.item_user_ratings.keys())
        
        # Items to predict
        predict_items = all_items - rated_items
        
        # Generate predictions
        predictions = []
        
        for item_id in predict_items:
            similar_items = self.item_similarity.get(item_id, {})
            if not similar_items:
                continue
                
            weighted_sum = 0.0
            similarity_sum = 0.0
            
            for rated_item, similarity in similar_items.items():
                if rated_item in user_ratings:
                    rating = user_ratings[rated_item]
                    weighted_sum += similarity * rating
                    similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                prediction = weighted_sum / similarity_sum
                # Clamp to valid rating range
                prediction = max(0.5, min(5.0, prediction))
                predictions.append((item_id, prediction))
        
        # Sort and return top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:limit]
