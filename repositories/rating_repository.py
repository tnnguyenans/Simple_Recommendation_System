#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rating repository for data access.
"""

import os
import csv
import logging
from typing import List, Dict, Optional
from datetime import datetime
from models.rating import Rating


class RatingRepository:
    """
    Repository for Rating data access.
    Handles loading, saving, and querying rating data.
    """
    
    def __init__(self, data_path: str = './data'):
        """
        Initialize the rating repository.
        
        Args:
            data_path: Path to directory containing data files
        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.data_path = data_path
        self.ratings_file = os.path.join(data_path, 'ratings.csv')
        self.ratings: Dict[int, Rating] = {}
        
        # Create data directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)
        
        # Load data if file exists
        if os.path.exists(self.ratings_file):
            self._load_ratings()
    
    def _load_ratings(self) -> None:
        """Load ratings from CSV file."""
        try:
            self._logger.info(f"Loading ratings from {self.ratings_file}")
            self.ratings = {}
            
            with open(self.ratings_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    rating_id = int(row['id'])
                    user_id = int(row['user_id'])
                    item_id = int(row['item_id'])
                    value = float(row['value'])
                    
                    # Parse timestamp if present
                    timestamp = None
                    if row.get('timestamp'):
                        try:
                            timestamp = datetime.fromisoformat(row['timestamp'])
                        except ValueError:
                            self._logger.warning(f"Invalid timestamp format for rating {rating_id}")
                    
                    # Create rating object
                    rating = Rating(
                        user_id=user_id,
                        item_id=item_id,
                        value=value,
                        id=rating_id,
                        timestamp=timestamp
                    )
                    self.ratings[rating_id] = rating
                    
            self._logger.info(f"Loaded {len(self.ratings)} ratings")
        
        except Exception as e:
            self._logger.error(f"Error loading ratings: {e}")
            # Initialize empty dictionary if there's an error
            self.ratings = {}
    
    def save_all(self) -> None:
        """Save all ratings to CSV file."""
        try:
            self._logger.info(f"Saving ratings to {self.ratings_file}")
            
            with open(self.ratings_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'user_id', 'item_id', 'value', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for rating in self.ratings.values():
                    timestamp_str = ''
                    if rating.timestamp:
                        timestamp_str = rating.timestamp.isoformat()
                        
                    writer.writerow({
                        'id': rating.id,
                        'user_id': rating.user_id,
                        'item_id': rating.item_id,
                        'value': rating.value,
                        'timestamp': timestamp_str
                    })
            
            self._logger.info(f"Saved {len(self.ratings)} ratings")
        
        except Exception as e:
            self._logger.error(f"Error saving ratings: {e}")
    
    def get_all(self) -> List[Rating]:
        """
        Get all ratings.
        
        Returns:
            List of all Rating objects
        """
        return list(self.ratings.values())
    
    def get_by_id(self, rating_id: int) -> Optional[Rating]:
        """
        Get rating by ID.
        
        Args:
            rating_id: Rating ID
            
        Returns:
            Rating object if found, None otherwise
        """
        rating = self.ratings.get(rating_id)
        if not rating:
            self._logger.debug(f"Rating not found: {rating_id}")
        return rating
    
    def get_by_user_id(self, user_id: int) -> List[Rating]:
        """
        Get ratings by user ID.
        
        Args:
            user_id: User ID
            
        Returns:
            List of ratings for the specified user
        """
        return [
            rating for rating in self.ratings.values()
            if rating.user_id == user_id
        ]
    
    def get_by_item_id(self, item_id: int) -> List[Rating]:
        """
        Get ratings by item ID.
        
        Args:
            item_id: Item ID
            
        Returns:
            List of ratings for the specified item
        """
        return [
            rating for rating in self.ratings.values()
            if rating.item_id == item_id
        ]
    
    def get_by_user_and_item(self, user_id: int, item_id: int) -> Optional[Rating]:
        """
        Get rating by user and item IDs.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Rating object if found, None otherwise
        """
        for rating in self.ratings.values():
            if rating.user_id == user_id and rating.item_id == item_id:
                return rating
        return None
    
    def save(self, rating: Rating) -> Rating:
        """
        Save a rating.
        
        Args:
            rating: Rating object to save
            
        Returns:
            Saved Rating object with ID assigned if it was None
        """
        # Check if there's an existing rating for this user and item
        existing = self.get_by_user_and_item(rating.user_id, rating.item_id)
        if existing:
            # Update the existing rating
            self._logger.debug(f"Updating existing rating for user {rating.user_id} and item {rating.item_id}")
            existing.value = rating.value
            existing.timestamp = rating.timestamp or datetime.now()
            rating = existing
        else:
            # Assign ID if not present
            if rating.id is None:
                rating.id = self._get_next_id()
            
            self._logger.debug(f"Saving new rating with ID {rating.id}")
            self.ratings[rating.id] = rating
        
        # Save to file
        self.save_all()
        
        return rating
    
    def delete(self, rating_id: int) -> bool:
        """
        Delete rating by ID.
        
        Args:
            rating_id: Rating ID
            
        Returns:
            True if deleted, False if not found
        """
        if rating_id in self.ratings:
            self._logger.debug(f"Deleting rating {rating_id}")
            del self.ratings[rating_id]
            self.save_all()
            return True
        else:
            self._logger.debug(f"Rating not found for deletion: {rating_id}")
            return False
    
    def _get_next_id(self) -> int:
        """
        Generate next rating ID.
        
        Returns:
            Next available rating ID
        """
        if not self.ratings:
            return 1
        return max(self.ratings.keys()) + 1
