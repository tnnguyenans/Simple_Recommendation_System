#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Item model for the recommendation system.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class Item:
    """
    Item domain model with essential validation.
    """
    name: str
    id: Optional[int] = None
    categories: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate on creation with logging."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self._logger.debug(f"Creating Item with name='{self.name}', id='{self.id}'")
        
        if not self.name:
            self._logger.error(f"Invalid item name: empty name not allowed")
            raise ValueError("Item name cannot be empty")
        
        self._logger.info(f"Item created successfully: {self.name}")
    
    def add_category(self, category: str) -> None:
        """
        Add a category to this item.
        
        Args:
            category: Category name
        """
        self._logger.debug(f"Adding category {category} to item {self.name}")
        if category not in self.categories:
            self.categories.append(category)
    
    def add_feature(self, feature_name: str, value: float) -> None:
        """
        Add or update a feature value for this item.
        
        Args:
            feature_name: Name of the feature
            value: Numeric value for the feature
        """
        self._logger.debug(f"Adding feature {feature_name}={value} for item {self.name}")
        self.features[feature_name] = value
    
    def get_feature_vector(self, feature_names: List[str]) -> List[float]:
        """
        Get feature values as a vector in the specified order.
        
        Args:
            feature_names: List of feature names in desired order
            
        Returns:
            List of feature values (0 if feature not present)
        """
        return [self.features.get(name, 0.0) for name in feature_names]


class ItemBuilder:
    """
    Builder for creating Item objects with complex setup.
    
    Design Pattern: Builder pattern for step-by-step construction.
    Use when Item has many optional parameters or complex feature sets.
    """
    
    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.debug("ItemBuilder initialized")
        
        self._name: str = ""
        self._id: Optional[int] = None
        self._categories: List[str] = []
        self._features: Dict[str, float] = {}
    
    def name(self, name: str) -> 'ItemBuilder':
        """Set the item name."""
        self._name = name
        return self
    
    def id(self, item_id: int) -> 'ItemBuilder':
        """Set the item ID."""
        self._id = item_id
        return self
    
    def category(self, category: str) -> 'ItemBuilder':
        """Add a category."""
        if category not in self._categories:
            self._categories.append(category)
        return self
    
    def feature(self, name: str, value: float) -> 'ItemBuilder':
        """Add a feature."""
        self._features[name] = value
        return self
    
    def build(self) -> Item:
        """
        Build and return the Item object.
        
        Returns:
            A fully configured Item object
        """
        item = Item(
            name=self._name,
            id=self._id,
            categories=self._categories,
            features=self._features
        )
        self._logger.info(f"Built item: {item.name}")
        return item
