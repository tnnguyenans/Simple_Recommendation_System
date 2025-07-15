#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
User model for the recommendation system.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class User:
    """
    User domain model with essential validation.
    """
    username: str
    id: Optional[int] = None
    preferences: Dict[str, float] = field(default_factory=dict)
    history: List[int] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate on creation with logging."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self._logger.debug(f"Creating User with username='{self.username}', id='{self.id}'")
        
        if not self.username or len(self.username) < 3:
            self._logger.error(f"Invalid username length: '{self.username}' (must be 3+ characters)")
            raise ValueError("Username must be 3+ characters")
        
        self._logger.info(f"User created successfully: {self.username}")
    
    def add_preference(self, category: str, weight: float) -> None:
        """
        Add or update a category preference for this user.
        
        Args:
            category: Category name
            weight: Preference weight (0-1)
        """
        if weight < 0 or weight > 1:
            self._logger.error(f"Invalid preference weight: {weight}")
            raise ValueError("Preference weight must be between 0 and 1")
        
        self._logger.debug(f"Adding preference {category}={weight} for user {self.username}")
        self.preferences[category] = weight
    
    def add_to_history(self, item_id: int) -> None:
        """
        Add an item to the user's history.
        
        Args:
            item_id: ID of the item the user interacted with
        """
        self._logger.debug(f"Adding item {item_id} to history for user {self.username}")
        if item_id not in self.history:
            self.history.append(item_id)


class UserBuilder:
    """
    Builder for creating User objects with complex setup.
    
    Design Pattern: Builder pattern for step-by-step construction.
    Use when User has many optional parameters or complex validation.
    """
    
    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.debug("UserBuilder initialized")
        
        self._username: str = ""
        self._id: Optional[int] = None
        self._preferences: Dict[str, float] = {}
        self._history: List[int] = []
    
    def username(self, username: str) -> 'UserBuilder':
        """Set the username."""
        self._username = username
        return self
    
    def id(self, user_id: int) -> 'UserBuilder':
        """Set the user ID."""
        self._id = user_id
        return self
    
    def preference(self, category: str, weight: float) -> 'UserBuilder':
        """Add a preference."""
        self._preferences[category] = weight
        return self
    
    def history_item(self, item_id: int) -> 'UserBuilder':
        """Add an item to history."""
        self._history.append(item_id)
        return self
    
    def build(self) -> User:
        """
        Build and return the User object.
        
        Returns:
            A fully configured User object
        """
        user = User(
            username=self._username,
            id=self._id,
            preferences=self._preferences,
            history=self._history
        )
        self._logger.info(f"Built user: {user.username}")
        return user
