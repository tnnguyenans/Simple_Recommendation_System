#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for collaborative filtering service.
"""

import os
import sys
import unittest
import numpy as np

# Add project root to path when running this file directly
if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)

from models.rating import Rating
from services.collaborative_filtering import CollaborativeFiltering


class TestCollaborativeFiltering(unittest.TestCase):
    """Test cases for CollaborativeFiltering algorithm."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self._logger = unittest.TestCase.shortDescription
        # Create algorithm instances for testing
        self.user_cf = CollaborativeFiltering(method='user-based')
        self.item_cf = CollaborativeFiltering(method='item-based')
        
        # Create test data
        self.ratings = [
            Rating(user_id=1, item_id=1, value=5.0, id=1),
            Rating(user_id=1, item_id=2, value=4.0, id=2),
            Rating(user_id=1, item_id=3, value=2.0, id=3),
            Rating(user_id=2, item_id=1, value=3.0, id=4),
            Rating(user_id=2, item_id=2, value=4.0, id=5),
            Rating(user_id=2, item_id=4, value=5.0, id=6),
            Rating(user_id=3, item_id=1, value=4.0, id=7),
            Rating(user_id=3, item_id=3, value=3.0, id=8),
            Rating(user_id=3, item_id=4, value=4.0, id=9),
        ]
        
        self.training_data = {
            'ratings': self.ratings,
            'users': [],  # Not used in basic CF
            'items': []   # Not used in basic CF
        }
    
    def test_initialization(self):
        """Test initialization of collaborative filtering."""
        self.assertEqual(self.user_cf.method, 'user-based')
        self.assertEqual(self.item_cf.method, 'item-based')
    
    def test_train_user_based(self):
        """Test training user-based collaborative filtering."""
        self.user_cf.train(self.training_data)
        
        # Check data structures
        self.assertEqual(len(self.user_cf.user_item_ratings), 3)
        self.assertEqual(len(self.user_cf.item_user_ratings), 4)
        
        # Check user means
        self.assertAlmostEqual(self.user_cf.user_means[1], 3.6667, places=4)
        
        # Check similarity computation
        self.assertIn(1, self.user_cf.user_similarity)
        self.assertIn(2, self.user_cf.user_similarity)
        self.assertIn(3, self.user_cf.user_similarity)
    
    def test_train_item_based(self):
        """Test training item-based collaborative filtering."""
        self.item_cf.train(self.training_data)
        
        # Check data structures
        self.assertEqual(len(self.item_cf.user_item_ratings), 3)
        self.assertEqual(len(self.item_cf.item_user_ratings), 4)
        
        # Check item means
        self.assertAlmostEqual(self.item_cf.item_means[1], 4.0, places=4)
        
        # Check similarity computation
        self.assertIn(1, self.item_cf.item_similarity)
        self.assertIn(2, self.item_cf.item_similarity)
        self.assertIn(3, self.item_cf.item_similarity)
        self.assertIn(4, self.item_cf.item_similarity)
    
    def test_user_based_recommendations(self):
        """Test user-based recommendations generation."""
        self.user_cf.train(self.training_data)
        
        # Get recommendations for user 1
        recs = self.user_cf.recommend_for_user(1)
        self.assertIsInstance(recs, list)
        
        # User 1 hasn't rated item 4, should be recommended
        item_ids = [item_id for item_id, _ in recs]
        self.assertIn(4, item_ids)
        
        # User 1 already rated items 1, 2, 3, shouldn't be recommended
        self.assertNotIn(1, item_ids)
        self.assertNotIn(2, item_ids)
        self.assertNotIn(3, item_ids)
    
    def test_item_based_recommendations(self):
        """Test item-based recommendations generation."""
        self.item_cf.train(self.training_data)
        
        # Get recommendations for user 1
        recs = self.item_cf.recommend_for_user(1)
        self.assertIsInstance(recs, list)
        
        # User 1 hasn't rated item 4, should be recommended
        item_ids = [item_id for item_id, _ in recs]
        self.assertIn(4, item_ids)
        
        # User 1 already rated items 1, 2, 3, shouldn't be recommended
        self.assertNotIn(1, item_ids)
        self.assertNotIn(2, item_ids)
        self.assertNotIn(3, item_ids)
    
    def test_empty_data(self):
        """Test behavior with empty data."""
        empty_data = {'ratings': [], 'users': [], 'items': []}
        
        # Should not raise exceptions
        self.user_cf.train(empty_data)
        self.item_cf.train(empty_data)
        
        # Should return empty recommendations
        self.assertEqual(self.user_cf.recommend_for_user(1), [])
        self.assertEqual(self.item_cf.recommend_for_user(1), [])


if __name__ == '__main__':
    unittest.main()
