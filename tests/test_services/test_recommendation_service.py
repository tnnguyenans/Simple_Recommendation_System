#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for main recommendation service.
"""

import os
import sys
import unittest
import logging
from unittest.mock import MagicMock, patch
import inspect

# Add project root to path to fix imports when running test directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from models.user import User
from models.item import Item
from models.rating import Rating
from services.recommendation_service import RecommendationService


class TestRecommendationService(unittest.TestCase):
    """Test cases for RecommendationService."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Configure logging for tests
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Create mock repositories
        self.user_repo = MagicMock()
        self.item_repo = MagicMock()
        self.rating_repo = MagicMock()
        
        # Configure repository mocks to return empty lists for get_all() calls
        # These will be used during RecommendationService initialization
        self.user_repo.get_all.return_value = []
        self.item_repo.get_all.return_value = []
        self.rating_repo.get_all.return_value = []
        
        # Create test data
        self.users = [
            User(username="user1", id=1),
            User(username="user2", id=2)
        ]
        
        self.items = [
            Item(name="Item 1", id=1, categories=["action"]),
            Item(name="Item 2", id=2, categories=["comedy"]),
            Item(name="Item 3", id=3, categories=["drama"])
        ]
        
        self.ratings = [
            Rating(user_id=1, item_id=1, value=5.0, id=1),
            Rating(user_id=1, item_id=2, value=3.0, id=2),
            Rating(user_id=2, item_id=2, value=4.0, id=3)
        ]
        
        # Set up repository returns
        self.user_repo.get_all.return_value = self.users
        self.item_repo.get_all.return_value = self.items
        self.rating_repo.get_all.return_value = self.ratings
        self.item_repo.get_by_id.side_effect = lambda id: next((i for i in self.items if i.id == id), None)
    
    @patch('services.recommendation_service.CollaborativeFiltering')
    def test_initialization_collaborative(self, mock_cf_class):
        """Test initialization with collaborative filtering."""
        # Set up mock with a properly tracked train method
        mock_cf = MagicMock()
        mock_cf_class.return_value = mock_cf
        
        # Create service with mocked repositories
        service = RecommendationService(
            user_repository=self.user_repo,
            item_repository=self.item_repo,
            rating_repository=self.rating_repo,
            algorithm_type='collaborative'
        )
        
        # Check algorithm initialization
        self.assertIn('collaborative', service.algorithms)
        self.assertNotIn('content-based', service.algorithms)
        
        # Check that the collaborative filtering class was instantiated
        mock_cf_class.assert_called_once()
        
        # Check that the train method was called on the collaborative filtering instance
        # No need to check call parameters or count as we just want to know it was called
        mock_cf.train.assert_called()
    
    @patch('services.recommendation_service.ContentBased')
    def test_initialization_content_based(self, mock_cb_class):
        """Test initialization with content-based filtering."""
        # Set up mock with a properly tracked train method
        mock_cb = MagicMock()
        mock_cb_class.return_value = mock_cb
        
        # Create service
        service = RecommendationService(
            user_repository=self.user_repo,
            item_repository=self.item_repo,
            rating_repository=self.rating_repo,
            algorithm_type='content-based'
        )
        
        # Check algorithm initialization
        self.assertIn('content-based', service.algorithms)
        self.assertNotIn('collaborative', service.algorithms)
        
        # Check that the content-based class was instantiated
        mock_cb_class.assert_called_once()
        
        # Check that the train method was called on the content-based instance
        mock_cb.train.assert_called()
    
    @patch('services.recommendation_service.CollaborativeFiltering')
    @patch('services.recommendation_service.ContentBased')
    def test_initialization_hybrid(self, mock_cb_class, mock_cf_class):
        """Test initialization with hybrid filtering."""
        # Set up mocks with properly tracked train methods
        mock_cf = MagicMock()
        mock_cb = MagicMock()
        mock_cf_class.return_value = mock_cf
        mock_cb_class.return_value = mock_cb
        
        # Create service
        service = RecommendationService(
            user_repository=self.user_repo,
            item_repository=self.item_repo,
            rating_repository=self.rating_repo,
            algorithm_type='hybrid'
        )
        
        # Check algorithm initialization
        self.assertIn('collaborative', service.algorithms)
        self.assertIn('content-based', service.algorithms)
        
        # Check that both algorithm classes were instantiated
        mock_cf_class.assert_called_once()
        mock_cb_class.assert_called_once()
        
        # Check that the train methods were called on both instances
        mock_cf.train.assert_called()
        mock_cb.train.assert_called()
    
    def test_get_recommendations(self):
        """Test getting recommendations for a user."""
        # Create mock algorithms
        mock_algo = MagicMock()
        mock_algo.recommend_for_user.return_value = [(3, 0.9), (1, 0.8)]
        
        # Create service with single mock algorithm
        service = RecommendationService(
            user_repository=self.user_repo,
            item_repository=self.item_repo,
            rating_repository=self.rating_repo
        )
        service.algorithms = {'mock': mock_algo}
        
        # Set up rating repository to return user's ratings
        self.rating_repo.get_by_user_id.return_value = [
            Rating(user_id=1, item_id=2, value=4.0)
        ]
        
        # Get recommendations
        recommendations = service.get_recommendations_for_user(
            user_id=1, 
            limit=2,
            exclude_rated=True
        )
        
        # Check results
        self.assertEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0]['item_id'], 3)
        self.assertEqual(recommendations[1]['item_id'], 1)
        
        # Verify repository calls
        self.rating_repo.get_by_user_id.assert_called_with(1)
        mock_algo.recommend_for_user.assert_called_with(1, limit=4)  # 2*limit
    
    def test_refresh_models(self):
        """Test refreshing recommendation models."""
        # Create mock algorithms
        mock_algo1 = MagicMock()
        mock_algo2 = MagicMock()
        
        # Create service with mock algorithms
        service = RecommendationService(
            user_repository=self.user_repo,
            item_repository=self.item_repo,
            rating_repository=self.rating_repo
        )
        service.algorithms = {
            'algo1': mock_algo1,
            'algo2': mock_algo2
        }
        
        # Refresh models
        service.refresh_models()
        
        # Verify training calls
        mock_algo1.train.assert_called_once()
        mock_algo2.train.assert_called_once()
        
        # Verify repository calls
        self.user_repo.get_all.assert_called()
        self.item_repo.get_all.assert_called()
        self.rating_repo.get_all.assert_called()


if __name__ == '__main__':
    unittest.main()
