#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for UserRepository.
"""

import os
import sys
import unittest
import tempfile
import shutil
import logging

# Add project root to path when running this file directly
if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)

from models.user import User
from repositories.user_repository import UserRepository


class TestUserRepository(unittest.TestCase):
    """Test cases for UserRepository."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Configure logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Create a temporary directory for test data
        self.test_data_dir = tempfile.mkdtemp()
        self.logger.debug(f"Created test directory: {self.test_data_dir}")
        
        # Create repository instance
        self.repository = UserRepository(self.test_data_dir)
        
        # Create some test users
        self.test_users = [
            User(username="test_user1", id=1, preferences={"action": 0.8}),
            User(username="test_user2", id=2, history=[1, 2, 3]),
            User(username="test_user3", id=3, preferences={"comedy": 0.5}, history=[4])
        ]
        
        # Add users to repository
        for user in self.test_users:
            self.repository.users[user.id] = user
        
        # Save to file
        self.repository.save_all()
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_data_dir)
        self.logger.debug(f"Removed test directory: {self.test_data_dir}")
    
    def test_initialization(self):
        """Test repository initialization."""
        # Test with existing data directory
        repo = UserRepository(self.test_data_dir)
        self.assertEqual(len(repo.users), 3)
        
        # Test with non-existing data directory
        new_dir = os.path.join(self.test_data_dir, 'new_dir')
        repo = UserRepository(new_dir)
        self.assertEqual(len(repo.users), 0)
        self.assertTrue(os.path.exists(new_dir))
    
    def test_get_all(self):
        """Test getting all users."""
        users = self.repository.get_all()
        self.assertEqual(len(users), 3)
        # Check if all test users are included
        user_ids = [user.id for user in users]
        for test_user in self.test_users:
            self.assertIn(test_user.id, user_ids)
    
    def test_get_by_id(self):
        """Test getting user by ID."""
        # Test existing user
        user = self.repository.get_by_id(1)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "test_user1")
        
        # Test non-existing user
        user = self.repository.get_by_id(999)
        self.assertIsNone(user)
    
    def test_save_new_user(self):
        """Test saving a new user."""
        # Create new user without ID
        new_user = User(username="new_user")
        saved_user = self.repository.save(new_user)
        
        # Check if ID was assigned
        self.assertIsNotNone(saved_user.id)
        
        # Check if user was added to repository
        self.assertIn(saved_user.id, self.repository.users)
        
        # Check if user was saved to file by creating a new repository instance
        new_repo = UserRepository(self.test_data_dir)
        loaded_user = new_repo.get_by_id(saved_user.id)
        self.assertIsNotNone(loaded_user)
        self.assertEqual(loaded_user.username, "new_user")
    
    def test_save_existing_user(self):
        """Test saving an existing user."""
        # Get existing user
        user = self.repository.get_by_id(1)
        
        # Modify user
        user.add_preference("comedy", 0.7)
        user.add_to_history(5)
        
        # Save user
        saved_user = self.repository.save(user)
        
        # Check if user was updated in repository
        self.assertEqual(saved_user.preferences["comedy"], 0.7)
        self.assertIn(5, saved_user.history)
        
        # Check if user was saved to file by creating a new repository instance
        new_repo = UserRepository(self.test_data_dir)
        loaded_user = new_repo.get_by_id(1)
        self.assertIsNotNone(loaded_user)
        self.assertEqual(loaded_user.preferences["comedy"], 0.7)
        self.assertIn(5, loaded_user.history)
    
    def test_delete(self):
        """Test deleting a user."""
        # Delete existing user
        result = self.repository.delete(1)
        self.assertTrue(result)
        self.assertNotIn(1, self.repository.users)
        
        # Check if user was deleted from file by creating a new repository instance
        new_repo = UserRepository(self.test_data_dir)
        self.assertIsNone(new_repo.get_by_id(1))
        
        # Try to delete non-existing user
        result = self.repository.delete(999)
        self.assertFalse(result)
    
    def test_get_next_id(self):
        """Test generating next user ID."""
        # With existing users
        next_id = self.repository._get_next_id()
        self.assertEqual(next_id, 4)  # Max ID (3) + 1
        
        # With empty repository
        empty_repo = UserRepository(os.path.join(self.test_data_dir, 'empty'))
        next_id = empty_repo._get_next_id()
        self.assertEqual(next_id, 1)


if __name__ == '__main__':
    unittest.main()
