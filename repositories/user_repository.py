#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
User repository for data access.
"""

import os
import csv
import json
import logging
from typing import List, Dict, Optional
from models.user import User


class UserRepository:
    """
    Repository for User data access.
    Handles loading, saving, and querying user data.
    """
    
    def __init__(self, data_path: str = './data'):
        """
        Initialize the user repository.
        
        Args:
            data_path: Path to directory containing data files
        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.data_path = data_path
        self.users_file = os.path.join(data_path, 'users.csv')
        self.users: Dict[int, User] = {}
        
        # Create data directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)
        
        # Load data if file exists
        if os.path.exists(self.users_file):
            self._load_users()
    
    def _load_users(self) -> None:
        """Load users from CSV file."""
        try:
            self._logger.info(f"Loading users from {self.users_file}")
            self.users = {}
            
            with open(self.users_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    user_id = int(row['id'])
                    username = row['username']
                    
                    # Parse preferences and history if present
                    preferences = {}
                    if row.get('preferences'):
                        try:
                            preferences = json.loads(row['preferences'])
                        except json.JSONDecodeError:
                            self._logger.warning(f"Invalid preferences JSON for user {user_id}")
                    
                    history = []
                    if row.get('history'):
                        try:
                            history = [int(x) for x in json.loads(row['history'])]
                        except (json.JSONDecodeError, ValueError):
                            self._logger.warning(f"Invalid history JSON for user {user_id}")
                    
                    # Create user object
                    user = User(
                        username=username,
                        id=user_id,
                        preferences=preferences,
                        history=history
                    )
                    self.users[user_id] = user
                    
            self._logger.info(f"Loaded {len(self.users)} users")
        
        except Exception as e:
            self._logger.error(f"Error loading users: {e}")
            # Initialize empty dictionary if there's an error
            self.users = {}
    
    def save_all(self) -> None:
        """Save all users to CSV file."""
        try:
            self._logger.info(f"Saving users to {self.users_file}")
            
            with open(self.users_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'username', 'preferences', 'history']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for user in self.users.values():
                    writer.writerow({
                        'id': user.id,
                        'username': user.username,
                        'preferences': json.dumps(user.preferences),
                        'history': json.dumps(user.history)
                    })
            
            self._logger.info(f"Saved {len(self.users)} users")
        
        except Exception as e:
            self._logger.error(f"Error saving users: {e}")
    
    def get_all(self) -> List[User]:
        """
        Get all users.
        
        Returns:
            List of all User objects
        """
        return list(self.users.values())
    
    def get_by_id(self, user_id: int) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object if found, None otherwise
        """
        user = self.users.get(user_id)
        if not user:
            self._logger.debug(f"User not found: {user_id}")
        return user
    
    def save(self, user: User) -> User:
        """
        Save a user.
        
        Args:
            user: User object to save
            
        Returns:
            Saved User object with ID assigned if it was None
        """
        # Assign ID if not present
        if user.id is None:
            user.id = self._get_next_id()
        
        self._logger.debug(f"Saving user {user.username} with ID {user.id}")
        self.users[user.id] = user
        
        # Save to file
        self.save_all()
        
        return user
    
    def delete(self, user_id: int) -> bool:
        """
        Delete user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            True if deleted, False if not found
        """
        if user_id in self.users:
            self._logger.debug(f"Deleting user {user_id}")
            del self.users[user_id]
            self.save_all()
            return True
        else:
            self._logger.debug(f"User not found for deletion: {user_id}")
            return False
    
    def _get_next_id(self) -> int:
        """
        Generate next user ID.
        
        Returns:
            Next available user ID
        """
        if not self.users:
            return 1
        return max(self.users.keys()) + 1
