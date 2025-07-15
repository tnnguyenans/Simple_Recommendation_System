#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for RatingRepository.
"""

import os
import sys
import unittest
import tempfile
import shutil
import logging
from datetime import datetime

# Add project root to path when running this file directly
if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)

from models.rating import Rating
from repositories.rating_repository import RatingRepository


class TestRatingRepository(unittest.TestCase):
    """Test cases for RatingRepository."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Create a temporary directory for test data
        self.test_data_dir = tempfile.mkdtemp()
        self._logger.debug(f"Created test directory: {self.test_data_dir}")
        
        # Create repository instance
        self.repository = RatingRepository(self.test_data_dir)
        
        # Create some test ratings
        self.timestamp = datetime(2025, 1, 1, 12, 0, 0)
        self.test_ratings = [
            Rating(user_id=1, item_id=1, value=5.0, id=1, timestamp=self.timestamp),
            Rating(user_id=1, item_id=2, value=4.0, id=2, timestamp=self.timestamp),
            Rating(user_id=2, item_id=1, value=3.5, id=3, timestamp=self.timestamp),
            Rating(user_id=2, item_id=3, value=4.5, id=4, timestamp=self.timestamp)
        ]
        
        # Add ratings to repository
        for rating in self.test_ratings:
            self.repository.ratings[rating.id] = rating
        
        # Save to file
        self.repository.save_all()
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_data_dir)
        self._logger.debug(f"Removed test directory: {self.test_data_dir}")
    
    def test_initialization(self):
        """Test repository initialization."""
        # Test with existing data directory
        repo = RatingRepository(self.test_data_dir)
        self.assertEqual(len(repo.ratings), 4)
        
        # Test with non-existing data directory
        new_dir = os.path.join(self.test_data_dir, 'new_dir')
        repo = RatingRepository(new_dir)
        self.assertEqual(len(repo.ratings), 0)
        self.assertTrue(os.path.exists(new_dir))
    
    def test_get_all(self):
        """Test getting all ratings."""
        ratings = self.repository.get_all()
        self.assertEqual(len(ratings), 4)
        # Check if all test ratings are included
        rating_ids = [rating.id for rating in ratings]
        for test_rating in self.test_ratings:
            self.assertIn(test_rating.id, rating_ids)
    
    def test_get_by_id(self):
        """Test getting rating by ID."""
        # Test existing rating
        rating = self.repository.get_by_id(1)
        self.assertIsNotNone(rating)
        self.assertEqual(rating.user_id, 1)
        self.assertEqual(rating.item_id, 1)
        self.assertEqual(rating.value, 5.0)
        
        # Test non-existing rating
        rating = self.repository.get_by_id(999)
        self.assertIsNone(rating)
    
    def test_get_by_user_id(self):
        """Test getting ratings by user ID."""
        # Test with existing user
        ratings = self.repository.get_by_user_id(1)
        self.assertEqual(len(ratings), 2)
        item_ids = [rating.item_id for rating in ratings]
        self.assertIn(1, item_ids)
        self.assertIn(2, item_ids)
        
        # Test with non-existing user
        ratings = self.repository.get_by_user_id(999)
        self.assertEqual(len(ratings), 0)
    
    def test_get_by_item_id(self):
        """Test getting ratings by item ID."""
        # Test with existing item
        ratings = self.repository.get_by_item_id(1)
        self.assertEqual(len(ratings), 2)
        user_ids = [rating.user_id for rating in ratings]
        self.assertIn(1, user_ids)
        self.assertIn(2, user_ids)
        
        # Test with non-existing item
        ratings = self.repository.get_by_item_id(999)
        self.assertEqual(len(ratings), 0)
    
    def test_get_by_user_and_item(self):
        """Test getting rating by user and item ID."""
        # Test existing rating
        rating = self.repository.get_by_user_and_item(1, 1)
        self.assertIsNotNone(rating)
        self.assertEqual(rating.value, 5.0)
        
        # Test non-existing rating
        rating = self.repository.get_by_user_and_item(1, 999)
        self.assertIsNone(rating)
    
    def test_save_new_rating(self):
        """Test saving a new rating."""
        # Create new rating without ID
        new_rating = Rating(user_id=3, item_id=1, value=4.0, timestamp=self.timestamp)
        saved_rating = self.repository.save(new_rating)
        
        # Check if ID was assigned
        self.assertIsNotNone(saved_rating.id)
        
        # Check if rating was added to repository
        self.assertIn(saved_rating.id, self.repository.ratings)
        
        # Check if rating was saved to file by creating a new repository instance
        new_repo = RatingRepository(self.test_data_dir)
        loaded_rating = new_repo.get_by_id(saved_rating.id)
        self.assertIsNotNone(loaded_rating)
        self.assertEqual(loaded_rating.user_id, 3)
        self.assertEqual(loaded_rating.item_id, 1)
        self.assertEqual(loaded_rating.value, 4.0)
    
    def test_save_existing_rating_by_id(self):
        """Test updating an existing rating by ID."""
        # Get existing rating
        rating = self.repository.get_by_id(1)
        
        # Modify rating
        rating.value = 4.5
        
        # Save rating
        saved_rating = self.repository.save(rating)
        
        # Check if rating was updated in repository
        self.assertEqual(saved_rating.value, 4.5)
        
        # Check if rating was saved to file by creating a new repository instance
        new_repo = RatingRepository(self.test_data_dir)
        loaded_rating = new_repo.get_by_id(1)
        self.assertIsNotNone(loaded_rating)
        self.assertEqual(loaded_rating.value, 4.5)
    
    def test_save_existing_rating_by_user_and_item(self):
        """Test updating an existing rating by user and item."""
        # Create new rating for same user and item as an existing one
        new_rating = Rating(user_id=1, item_id=1, value=3.5)
        saved_rating = self.repository.save(new_rating)
        
        # Check if the existing rating was updated
        self.assertEqual(saved_rating.id, 1)  # Should have same ID as existing
        self.assertEqual(saved_rating.value, 3.5)
        
        # Check if rating was saved to file by creating a new repository instance
        new_repo = RatingRepository(self.test_data_dir)
        loaded_rating = new_repo.get_by_id(1)
        self.assertIsNotNone(loaded_rating)
        self.assertEqual(loaded_rating.value, 3.5)
    
    def test_delete(self):
        """Test deleting a rating."""
        # Delete existing rating
        result = self.repository.delete(1)
        self.assertTrue(result)
        self.assertNotIn(1, self.repository.ratings)
        
        # Check if rating was deleted from file by creating a new repository instance
        new_repo = RatingRepository(self.test_data_dir)
        self.assertIsNone(new_repo.get_by_id(1))
        
        # Try to delete non-existing rating
        result = self.repository.delete(999)
        self.assertFalse(result)
    
    def test_get_next_id(self):
        """Test generating next rating ID."""
        # With existing ratings
        next_id = self.repository._get_next_id()
        self.assertEqual(next_id, 5)  # Max ID (4) + 1
        
        # With empty repository
        empty_repo = RatingRepository(os.path.join(self.test_data_dir, 'empty'))
        next_id = empty_repo._get_next_id()
        self.assertEqual(next_id, 1)


if __name__ == '__main__':
    unittest.main()
