#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for Rating model.
"""

import os
import sys
import unittest
from datetime import datetime

# Add project root to path when running this file directly
if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)

from models.rating import Rating

class TestRating(unittest.TestCase):
    """Test cases for Rating model."""
    
    def test_rating_creation(self):
        """Test basic rating creation."""
        rating = Rating(user_id=1, item_id=2, value=4.5)
        self.assertEqual(rating.user_id, 1)
        self.assertEqual(rating.item_id, 2)
        self.assertEqual(rating.value, 4.5)
        self.assertIsNone(rating.id)
        self.assertIsNotNone(rating.timestamp)  # Should default to now
    
    def test_rating_with_timestamp(self):
        """Test rating creation with timestamp."""
        timestamp = datetime(2025, 1, 1, 12, 0, 0)
        rating = Rating(user_id=1, item_id=2, value=4.0, timestamp=timestamp)
        self.assertEqual(rating.timestamp, timestamp)
    
    def test_rating_validation(self):
        """Test rating validation."""
        # Rating value must be between 0 and 5
        with self.assertRaises(ValueError):
            Rating(user_id=1, item_id=2, value=-1)  # Too low
            
        with self.assertRaises(ValueError):
            Rating(user_id=1, item_id=2, value=6)  # Too high


if __name__ == '__main__':
    unittest.main()
