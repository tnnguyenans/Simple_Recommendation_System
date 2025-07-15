#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main recommendation service that coordinates different recommendation algorithms.
"""

import logging
from typing import List, Dict, Any, Protocol, Optional
from services.collaborative_filtering import CollaborativeFiltering
from services.content_based import ContentBased


class RecommendationAlgorithm(Protocol):
    """Protocol defining the interface for recommendation algorithms."""
    
    def train(self, data: Dict[str, Any]) -> None:
        """Train the recommendation algorithm with data."""
        ...
    
    def recommend_for_user(self, user_id: int, limit: int = 10) -> List[tuple[int, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID to recommend for
            limit: Maximum number of recommendations
            
        Returns:
            List of tuples (item_id, predicted_rating)
        """
        ...


class RecommendationService:
    """
    Main recommendation service that coordinates different recommendation algorithms.
    """
    
    def __init__(
        self,
        user_repository: Any,
        item_repository: Any,
        rating_repository: Any,
        algorithm_type: str = 'collaborative'
    ):
        """
        Initialize the recommendation service.
        
        Args:
            user_repository: Repository for user data
            item_repository: Repository for item data
            rating_repository: Repository for rating data
            algorithm_type: Type of algorithm to use ('collaborative', 'content-based', or 'hybrid')
        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.info(f"Initializing recommendation service with algorithm: {algorithm_type}")
        
        self.user_repository = user_repository
        self.item_repository = item_repository
        self.rating_repository = rating_repository
        self.algorithm_type = algorithm_type
        
        # Initialize the appropriate algorithm(s)
        self.algorithms: Dict[str, RecommendationAlgorithm] = {}
        
        if algorithm_type in ['collaborative', 'hybrid']:
            self.algorithms['collaborative'] = CollaborativeFiltering()
            self._logger.info("Collaborative filtering algorithm initialized")
            
        if algorithm_type in ['content-based', 'hybrid']:
            self.algorithms['content-based'] = ContentBased()
            self._logger.info("Content-based algorithm initialized")
        
        # Train algorithms with initial data
        self._train_algorithms()
    
    def _train_algorithms(self) -> None:
        """Train all active recommendation algorithms with current data."""
        try:
            self._logger.info("Training recommendation algorithms")
            
            # Prepare training data
            training_data = {
                'users': self.user_repository.get_all(),
                'items': self.item_repository.get_all(),
                'ratings': self.rating_repository.get_all()
            }
            
            # Train each algorithm
            for name, algorithm in self.algorithms.items():
                self._logger.debug(f"Training {name} algorithm")
                algorithm.train(training_data)
                
            self._logger.info("Algorithm training completed")
            
        except Exception as e:
            self._logger.error(f"Error training algorithms: {e}")
            raise
    
    def refresh_models(self) -> None:
        """Refresh all recommendation models with latest data."""
        self._logger.info("Refreshing recommendation models")
        self._train_algorithms()
    
    def get_recommendations_for_user(
        self, 
        user_id: int, 
        limit: int = 10,
        exclude_rated: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for a specific user.
        
        Args:
            user_id: ID of the user to get recommendations for
            limit: Maximum number of recommendations to return
            exclude_rated: Whether to exclude items the user has already rated
            
        Returns:
            List of recommended items with scores
        """
        self._logger.info(f"Getting recommendations for user {user_id}, limit={limit}")
        
        try:
            # Get user's existing ratings if needed
            user_rated_items = set()
            if exclude_rated:
                user_ratings = self.rating_repository.get_by_user_id(user_id)
                user_rated_items = {r.item_id for r in user_ratings}
                self._logger.debug(f"User {user_id} has rated {len(user_rated_items)} items")
            
            # Get recommendations from each active algorithm
            all_recommendations: Dict[int, float] = {}
            
            for name, algorithm in self.algorithms.items():
                self._logger.debug(f"Getting recommendations from {name} algorithm")
                algorithm_recs = algorithm.recommend_for_user(user_id, limit=limit * 2)
                
                # Add to combined recommendations with appropriate weighting
                weight = 1.0 / len(self.algorithms)  # Equal weighting by default
                for item_id, score in algorithm_recs:
                    if exclude_rated and item_id in user_rated_items:
                        continue
                    
                    if item_id in all_recommendations:
                        all_recommendations[item_id] += score * weight
                    else:
                        all_recommendations[item_id] = score * weight
            
            # Sort and limit results
            sorted_recommendations = sorted(
                all_recommendations.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            # Convert to list of dicts with item details
            result = []
            for item_id, score in sorted_recommendations:
                item = self.item_repository.get_by_id(item_id)
                if item:
                    result.append({
                        'item_id': item_id,
                        'name': item.name,
                        'score': round(score, 3),
                        'categories': item.categories
                    })
            
            self._logger.info(f"Returning {len(result)} recommendations for user {user_id}")
            return result
            
        except Exception as e:
            self._logger.error(f"Error getting recommendations: {e}")
            raise
