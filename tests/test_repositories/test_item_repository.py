#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for ItemRepository.
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

from models.item import Item
from repositories.item_repository import ItemRepository


class TestItemRepository(unittest.TestCase):
    """Test cases for ItemRepository."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Configure logging
        logging.basicConfig(level=logging.DEBUG)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Create a temporary directory for test data
        self.test_data_dir = tempfile.mkdtemp()
        self._logger.debug(f"Created test directory: {self.test_data_dir}")
        
        # Create repository instance
        self.repository = ItemRepository(self.test_data_dir)
        
        # Create some test items
        self.test_items = [
            Item(name="Movie 1", id=1, categories=["action", "sci-fi"], 
                 features={"length": 120, "year": 2020}),
            Item(name="Movie 2", id=2, categories=["comedy", "romance"], 
                 features={"length": 95, "year": 2021}),
            Item(name="Movie 3", id=3, categories=["action", "thriller"], 
                 features={"length": 130, "year": 2019})
        ]
        
        # Add items to repository
        for item in self.test_items:
            self.repository.items[item.id] = item
        
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
        repo = ItemRepository(self.test_data_dir)
        self.assertEqual(len(repo.items), 3)
        
        # Test with non-existing data directory
        new_dir = os.path.join(self.test_data_dir, 'new_dir')
        repo = ItemRepository(new_dir)
        self.assertEqual(len(repo.items), 0)
        self.assertTrue(os.path.exists(new_dir))
    
    def test_get_all(self):
        """Test getting all items."""
        items = self.repository.get_all()
        self.assertEqual(len(items), 3)
        # Check if all test items are included
        item_ids = [item.id for item in items]
        for test_item in self.test_items:
            self.assertIn(test_item.id, item_ids)
    
    def test_get_by_id(self):
        """Test getting item by ID."""
        # Test existing item
        item = self.repository.get_by_id(1)
        self.assertIsNotNone(item)
        self.assertEqual(item.name, "Movie 1")
        
        # Test non-existing item
        item = self.repository.get_by_id(999)
        self.assertIsNone(item)
    
    def test_get_by_category(self):
        """Test getting items by category."""
        # Test with existing category
        items = self.repository.get_by_category("action")
        self.assertEqual(len(items), 2)
        names = [item.name for item in items]
        self.assertIn("Movie 1", names)
        self.assertIn("Movie 3", names)
        
        # Test with non-existing category
        items = self.repository.get_by_category("horror")
        self.assertEqual(len(items), 0)
    
    def test_save_new_item(self):
        """Test saving a new item."""
        # Create new item without ID
        new_item = Item(name="New Movie", categories=["drama"])
        saved_item = self.repository.save(new_item)
        
        # Check if ID was assigned
        self.assertIsNotNone(saved_item.id)
        
        # Check if item was added to repository
        self.assertIn(saved_item.id, self.repository.items)
        
        # Check if item was saved to file by creating a new repository instance
        new_repo = ItemRepository(self.test_data_dir)
        loaded_item = new_repo.get_by_id(saved_item.id)
        self.assertIsNotNone(loaded_item)
        self.assertEqual(loaded_item.name, "New Movie")
        self.assertIn("drama", loaded_item.categories)
    
    def test_save_existing_item(self):
        """Test saving an existing item."""
        # Get existing item
        item = self.repository.get_by_id(1)
        
        # Modify item
        item.add_category("fantasy")
        item.add_feature("rating", 4.5)
        
        # Save item
        saved_item = self.repository.save(item)
        
        # Check if item was updated in repository
        self.assertIn("fantasy", saved_item.categories)
        self.assertEqual(saved_item.features["rating"], 4.5)
        
        # Check if item was saved to file by creating a new repository instance
        new_repo = ItemRepository(self.test_data_dir)
        loaded_item = new_repo.get_by_id(1)
        self.assertIsNotNone(loaded_item)
        self.assertIn("fantasy", loaded_item.categories)
        self.assertEqual(float(loaded_item.features["rating"]), 4.5)
    
    def test_delete(self):
        """Test deleting an item."""
        # Delete existing item
        result = self.repository.delete(1)
        self.assertTrue(result)
        self.assertNotIn(1, self.repository.items)
        
        # Check if item was deleted from file by creating a new repository instance
        new_repo = ItemRepository(self.test_data_dir)
        self.assertIsNone(new_repo.get_by_id(1))
        
        # Try to delete non-existing item
        result = self.repository.delete(999)
        self.assertFalse(result)
    
    def test_get_next_id(self):
        """Test generating next item ID."""
        # With existing items
        next_id = self.repository._get_next_id()
        self.assertEqual(next_id, 4)  # Max ID (3) + 1
        
        # With empty repository
        empty_repo = ItemRepository(os.path.join(self.test_data_dir, 'empty'))
        next_id = empty_repo._get_next_id()
        self.assertEqual(next_id, 1)


if __name__ == '__main__':
    unittest.main()
