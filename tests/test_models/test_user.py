#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for User model.
"""

import os
import sys
import unittest

# Add project root to path when running this file directly
if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)

from models.user import User, UserBuilder


class TestUser(unittest.TestCase):
    """Test cases for User model."""
    
    def test_user_creation(self):
        """Test basic user creation."""
        user = User(username="testuser")
        self.assertEqual(user.username, "testuser")
        self.assertIsNone(user.id)
        self.assertEqual(user.preferences, {})
        self.assertEqual(user.history, [])
    
    def test_user_validation(self):
        """Test user validation."""
        # Username must be at least 3 characters
        with self.assertRaises(ValueError):
            User(username="ab")  # Too short
    
    def test_add_preference(self):
        """Test adding preferences."""
        user = User(username="testuser")
        user.add_preference("action", 0.8)
        self.assertEqual(user.preferences["action"], 0.8)
        
        # Test preference validation
        with self.assertRaises(ValueError):
            user.add_preference("comedy", 1.5)  # Out of range
    
    def test_add_to_history(self):
        """Test adding items to history."""
        user = User(username="testuser")
        user.add_to_history(1)
        user.add_to_history(2)
        self.assertEqual(user.history, [1, 2])
        
        # Test duplicates aren't added
        user.add_to_history(1)
        self.assertEqual(user.history, [1, 2])
    
    def test_user_builder(self):
        """Test UserBuilder pattern."""
        user = (UserBuilder()
                .username("builder_test")
                .id(42)
                .preference("action", 0.7)
                .preference("comedy", 0.3)
                .history_item(1)
                .history_item(2)
                .build())
        
        self.assertEqual(user.username, "builder_test")
        self.assertEqual(user.id, 42)
        self.assertEqual(user.preferences, {"action": 0.7, "comedy": 0.3})
        self.assertEqual(user.history, [1, 2])


if __name__ == '__main__':
    unittest.main()
