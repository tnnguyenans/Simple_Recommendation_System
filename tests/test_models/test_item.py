#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for Item model.
"""

import os
import sys
import unittest

# Add project root to path when running this file directly
if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)

from models.item import Item, ItemBuilder


class TestItem(unittest.TestCase):
    """Test cases for Item model."""
    
    def test_item_creation(self):
        """Test basic item creation."""
        item = Item(name="Test Item")
        self.assertEqual(item.name, "Test Item")
        self.assertIsNone(item.id)
        self.assertEqual(item.categories, [])
        self.assertEqual(item.features, {})
    
    def test_item_validation(self):
        """Test item validation."""
        # Name cannot be empty
        with self.assertRaises(ValueError):
            Item(name="")  # Empty name
    
    def test_add_category(self):
        """Test adding categories."""
        item = Item(name="Test Item")
        item.add_category("action")
        item.add_category("comedy")
        self.assertEqual(item.categories, ["action", "comedy"])
        
        # Test duplicate categories aren't added
        item.add_category("action")
        self.assertEqual(item.categories, ["action", "comedy"])
    
    def test_add_feature(self):
        """Test adding features."""
        item = Item(name="Test Item")
        item.add_feature("length", 120.5)
        item.add_feature("rating", 4.5)
        self.assertEqual(item.features["length"], 120.5)
        self.assertEqual(item.features["rating"], 4.5)
    
    def test_get_feature_vector(self):
        """Test feature vector extraction."""
        item = Item(name="Test Item")
        item.add_feature("f1", 0.5)
        item.add_feature("f2", 1.0)
        
        # Test with all features present
        features = item.get_feature_vector(["f1", "f2"])
        self.assertEqual(features, [0.5, 1.0])
        
        # Test with missing features (should be 0)
        features = item.get_feature_vector(["f1", "f2", "f3"])
        self.assertEqual(features, [0.5, 1.0, 0.0])
        
        # Test with different order
        features = item.get_feature_vector(["f2", "f1"])
        self.assertEqual(features, [1.0, 0.5])
    
    def test_item_builder(self):
        """Test ItemBuilder pattern."""
        item = (ItemBuilder()
                .name("builder_test")
                .id(42)
                .category("action")
                .category("thriller")
                .feature("length", 115.5)
                .feature("rating", 4.2)
                .build())
        
        self.assertEqual(item.name, "builder_test")
        self.assertEqual(item.id, 42)
        self.assertEqual(item.categories, ["action", "thriller"])
        self.assertEqual(item.features["length"], 115.5)
        self.assertEqual(item.features["rating"], 4.2)


if __name__ == '__main__':
    unittest.main()
