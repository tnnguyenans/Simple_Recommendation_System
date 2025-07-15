#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rating model for the recommendation system.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class Rating:
    """
    Rating domain model with essential validation.
    """
    user_id: int
    item_id: int
    value: float
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Validate on creation with logging."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self._logger.debug(f"Creating Rating: user_id={self.user_id}, item_id={self.item_id}, value={self.value}")
        
        if self.value < 0 or self.value > 5:
            self._logger.error(f"Invalid rating value: {self.value}")
            raise ValueError("Rating must be between 0 and 5")
        
        if self.timestamp is None:
            self.timestamp = datetime.now()
            
        self._logger.info(f"Rating created successfully for user {self.user_id} on item {self.item_id}")
