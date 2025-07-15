#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Item repository for data access.
"""

import os
import csv
import json
import logging
from typing import List, Dict, Optional
from models.item import Item


class ItemRepository:
    """
    Repository for Item data access.
    Handles loading, saving, and querying item data.
    """
    
    def __init__(self, data_path: str = './data'):
        """
        Initialize the item repository.
        
        Args:
            data_path: Path to directory containing data files
        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.data_path = data_path
        self.items_file = os.path.join(data_path, 'items.csv')
        self.items: Dict[int, Item] = {}
        
        # Create data directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)
        
        # Load data if file exists
        if os.path.exists(self.items_file):
            self._load_items()
    
    def _load_items(self) -> None:
        """Load items from CSV file."""
        try:
            self._logger.info(f"Loading items from {self.items_file}")
            self.items = {}
            
            with open(self.items_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    item_id = int(row['id'])
                    name = row['name']
                    
                    # Parse categories and features if present
                    categories = []
                    if row.get('categories'):
                        try:
                            categories = json.loads(row['categories'])
                        except json.JSONDecodeError:
                            self._logger.warning(f"Invalid categories JSON for item {item_id}")
                    
                    features = {}
                    if row.get('features'):
                        try:
                            features = json.loads(row['features'])
                            # Convert string keys back to float values
                            features = {k: float(v) for k, v in features.items()}
                        except (json.JSONDecodeError, ValueError):
                            self._logger.warning(f"Invalid features JSON for item {item_id}")
                    
                    # Create item object
                    item = Item(
                        name=name,
                        id=item_id,
                        categories=categories,
                        features=features
                    )
                    self.items[item_id] = item
                    
            self._logger.info(f"Loaded {len(self.items)} items")
        
        except Exception as e:
            self._logger.error(f"Error loading items: {e}")
            # Initialize empty dictionary if there's an error
            self.items = {}
    
    def save_all(self) -> None:
        """Save all items to CSV file."""
        try:
            self._logger.info(f"Saving items to {self.items_file}")
            
            with open(self.items_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'name', 'categories', 'features']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for item in self.items.values():
                    writer.writerow({
                        'id': item.id,
                        'name': item.name,
                        'categories': json.dumps(item.categories),
                        'features': json.dumps({k: str(v) for k, v in item.features.items()})
                    })
            
            self._logger.info(f"Saved {len(self.items)} items")
        
        except Exception as e:
            self._logger.error(f"Error saving items: {e}")
    
    def get_all(self) -> List[Item]:
        """
        Get all items.
        
        Returns:
            List of all Item objects
        """
        return list(self.items.values())
    
    def get_by_id(self, item_id: int) -> Optional[Item]:
        """
        Get item by ID.
        
        Args:
            item_id: Item ID
            
        Returns:
            Item object if found, None otherwise
        """
        item = self.items.get(item_id)
        if not item:
            self._logger.debug(f"Item not found: {item_id}")
        return item
    
    def get_by_category(self, category: str) -> List[Item]:
        """
        Get items by category.
        
        Args:
            category: Category name
            
        Returns:
            List of items in the category
        """
        return [
            item for item in self.items.values()
            if category in item.categories
        ]
    
    def save(self, item: Item) -> Item:
        """
        Save an item.
        
        Args:
            item: Item object to save
            
        Returns:
            Saved Item object with ID assigned if it was None
        """
        # Assign ID if not present
        if item.id is None:
            item.id = self._get_next_id()
        
        self._logger.debug(f"Saving item {item.name} with ID {item.id}")
        self.items[item.id] = item
        
        # Save to file
        self.save_all()
        
        return item
    
    def delete(self, item_id: int) -> bool:
        """
        Delete item by ID.
        
        Args:
            item_id: Item ID
            
        Returns:
            True if deleted, False if not found
        """
        if item_id in self.items:
            self._logger.debug(f"Deleting item {item_id}")
            del self.items[item_id]
            self.save_all()
            return True
        else:
            self._logger.debug(f"Item not found for deletion: {item_id}")
            return False
    
    def _get_next_id(self) -> int:
        """
        Generate next item ID.
        
        Returns:
            Next available item ID
        """
        if not self.items:
            return 1
        return max(self.items.keys()) + 1
