#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the recommendation system application.
"""

import logging
import argparse
from models.user import User
from models.item import Item
from services.recommendation_service import RecommendationService
from repositories.user_repository import UserRepository
from repositories.item_repository import ItemRepository
from repositories.rating_repository import RatingRepository

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommendation_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main function to run the recommendation system."""
    logger.info("Starting recommendation system")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Recommendation System')
    parser.add_argument('--data-path', default='./data', 
                        help='Path to data directory')
    parser.add_argument('--algorithm', default='collaborative',
                        choices=['collaborative', 'content-based', 'hybrid'],
                        help='Recommendation algorithm to use')
    args = parser.parse_args()
    
    # Initialize repositories
    user_repo = UserRepository(args.data_path)
    item_repo = ItemRepository(args.data_path)
    rating_repo = RatingRepository(args.data_path)
    
    # Initialize recommendation service
    recommendation_service = RecommendationService(
        user_repository=user_repo,
        item_repository=item_repo,
        rating_repository=rating_repo,
        algorithm_type=args.algorithm
    )
    
    # Example: Get recommendations for a user
    try:
        user_id = 1  # Example user ID
        recommended_items = recommendation_service.get_recommendations_for_user(user_id, limit=5)
        logger.info(f"Top 5 recommendations for user {user_id}: {recommended_items}")
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
    
    logger.info("Recommendation system completed")


if __name__ == "__main__":
    main()
