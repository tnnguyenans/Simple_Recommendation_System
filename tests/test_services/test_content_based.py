#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for content-based filtering service.
"""

import os
import sys
import unittest
import logging

# Add project root to path when running this file directly
if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)

from models.user import User
from models.item import Item
from models.rating import Rating
from services.content_based import ContentBased


class TestContentBased(unittest.TestCase):
    """Test cases for ContentBased algorithm."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Configure logging for tests
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Create algorithm instance for testing
        self.content_based = ContentBased()
        
        # Create test data - users
        self.users = [
            User(username="user1", id=1, preferences={"action": 0.8, "sci-fi": 0.6}),
            User(username="user2", id=2, preferences={"comedy": 0.9, "romance": 0.7}),
            User(username="user3", id=3, preferences={"action": 0.5, "thriller": 0.8})
        ]
        
        # Create test data - items
        self.items = [
            Item(name="Movie 1", id=1, categories=["action", "sci-fi"], 
                 features={"length": 120, "year": 2020, "budget": 100}),
            Item(name="Movie 2", id=2, categories=["comedy", "romance"], 
                 features={"length": 95, "year": 2021, "budget": 50}),
            Item(name="Movie 3", id=3, categories=["action", "thriller"], 
                 features={"length": 130, "year": 2019, "budget": 150}),
            Item(name="Movie 4", id=4, categories=["sci-fi", "thriller"], 
                 features={"length": 140, "year": 2022, "budget": 200})
        ]
        
        # Create test data - ratings
        self.ratings = [
            Rating(user_id=1, item_id=1, value=5.0, id=1),
            Rating(user_id=1, item_id=3, value=4.0, id=2),
            Rating(user_id=2, item_id=2, value=5.0, id=3),
            Rating(user_id=3, item_id=3, value=4.5, id=4),
            Rating(user_id=3, item_id=4, value=4.0, id=5)
        ]
        
        self.training_data = {
            'users': self.users,
            'items': self.items,
            'ratings': self.ratings
        }
        
    def test_initialization(self):
        """Test initialization of content-based filtering."""
        self.logger.debug("Testing initialization")
        self.assertIsInstance(self.content_based, ContentBased)
        self.assertEqual(self.content_based.user_profiles, {})
        self.assertEqual(self.content_based.item_features, {})
        self.assertEqual(self.content_based.feature_list, [])
    
    def test_train(self):
        """Test training content-based model."""
        self.logger.debug("Testing training")
        self.content_based.train(self.training_data)
        
        # Check feature extraction
        self.assertGreater(len(self.content_based.feature_list), 0)
        
        # Check item features
        for item in self.items:
            self.assertIn(item.id, self.content_based.item_features)
            
        # Check user profiles
        for user_id in [1, 2, 3]:
            self.assertIn(user_id, self.content_based.user_profiles)
            self.assertGreater(len(self.content_based.user_profiles[user_id]), 0)
    
    def test_recommendations(self):
        """Test recommendation generation."""
        self.logger.debug("Testing recommendation generation")
        self.content_based.train(self.training_data)
        
        # Get recommendations for user 1
        recs = self.content_based.recommend_for_user(1)
        self.assertIsInstance(recs, list)
        
        # User 1 should have recommendations
        self.assertGreater(len(recs), 0)
        
        # Recommendations should be tuples of (item_id, score)
        for rec in recs:
            self.assertIsInstance(rec, tuple)
            self.assertEqual(len(rec), 2)
        
        # User 1 already rated items 1 and 3, shouldn't be recommended
        item_ids = [item_id for item_id, _ in recs]
        self.assertNotIn(1, item_ids)
        self.assertNotIn(3, item_ids)
        
        # Movies similar to user's preferences should be recommended
        # Movie 4 has sci-fi category which user 1 likes
        self.assertIn(4, item_ids)
    
    def test_explain_recommendation(self):
        """Test recommendation explanation."""
        self.logger.debug("Testing recommendation explanation")
        
        # Print training data to debug
        self.logger.debug(f"Items in training: {[item.name for item in self.training_data['items']]}")
        for item in self.training_data['items']:
            if item.id == 4:
                self.logger.debug(f"Item 4 categories: {item.categories}")
                self.logger.debug(f"Item 4 features: {item.features}")
        
        user1 = None
        for user in self.training_data['users']:
            if user.id == 1:
                user1 = user
                self.logger.debug(f"User 1 preferences: {user1.preferences}")
        
        # Train the model
        self.content_based.train(self.training_data)
        
        # Debug feature list and item features
        self.logger.debug(f"Feature list: {self.content_based.feature_list}")
        self.logger.debug(f"Item 4 features: {self.content_based.item_features.get(4, {})}")
        
        # Get explanation for why item 4 is recommended to user 1
        explanation = self.content_based.explain_recommendation(1, 4)
        
        # Should have explanation with features
        self.assertIn("top_features", explanation)
        self.assertIsInstance(explanation["top_features"], list)
        
        # Print the explanation to debug
        self.logger.debug(f"Recommendation explanation: {explanation}")
        
        # Instead of checking for sci-fi directly, we'll update the test to check for category_sci-fi
        # since that's how it's stored in the content-based model
        found_sci_fi = False
        all_feature_names = []
        for feature in explanation["top_features"]:
            feature_name = feature["feature"]
            all_feature_names.append(feature_name)
            if "sci-fi" in feature_name:
                found_sci_fi = True
                break
        
        # If not found, let's modify the assertion message to be more informative
        self.assertTrue(found_sci_fi, f"sci-fi should be in the explanation features. Found features: {all_feature_names}")
    
    def test_empty_data(self):
        """Test behavior with empty data."""
        self.logger.debug("Testing empty data handling")
        empty_data = {'ratings': [], 'users': [], 'items': []}
        
        # Should not raise exceptions
        self.content_based.train(empty_data)
        
        # Should return empty recommendations
        self.assertEqual(self.content_based.recommend_for_user(1), [])


if __name__ == '__main__':
    unittest.main()
