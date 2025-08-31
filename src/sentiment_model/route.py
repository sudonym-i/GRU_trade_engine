#!/usr/bin/env python3
"""
Route functions for Sentiment Model Web Scraping and Analysis

This module provides routing functions to orchestrate web scraping and sentiment analysis
for the neural trade engine. It integrates the C++ webscraper with Python-based
sentiment analysis using the BERT model.

Author: ML Trading Bot Project
Purpose: Web scraping and sentiment analysis pipeline for trading decisions
"""

import os
import subprocess
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from .model import BertSentimentModel, ModelConfig, SentimentPredictor
from .tokenize_pipeline import TokenizationPipeline, TokenizationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def pull_from_web() -> Dict[str, Union[bool, str, int]]:
    """
    Execute the webscraper to pull data from configured web sources.
    
    This function navigates to the web_scraper build directory and executes
    the compiled webscraper binary to collect fresh data.
    
    Returns:
        Dict: Status report containing success flag, message, and exit code
    """
    logger.info("Starting web scraping operation")
    
    # Define paths relative to current module location
    current_dir = Path(__file__).parent
    webscraper_build_dir = current_dir / "web_scraper" / "build"
    webscraper_exe = "webscrape.exe"
    
    try:
        # Verify webscraper directory exists
        if not webscraper_build_dir.exists():
            error_msg = f"Webscraper build directory not found: {webscraper_build_dir}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "exit_code": -1,
                "data_collected": False
            }
        
        # Verify executable exists
        webscraper_path = webscraper_build_dir / webscraper_exe
        if not webscraper_path.exists():
            error_msg = f"Webscraper executable not found: {webscraper_path}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "exit_code": -1,
                "data_collected": False
            }
        
        logger.info(f"Found webscraper at: {webscraper_path}")
        
        # Change to webscraper build directory and execute
        original_cwd = os.getcwd()
        
        try:
            os.chdir(webscraper_build_dir)
            logger.info(f"Changed to directory: {webscraper_build_dir}")
            
            # Make executable (ensure permissions)
            chmod_result = subprocess.run(
                ["chmod", "+x", webscraper_exe],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if chmod_result.returncode != 0:
                logger.warning(f"chmod warning: {chmod_result.stderr}")
            
            # Execute the webscraper
            logger.info("Executing webscraper...")
            result = subprocess.run(
                [f"./{webscraper_exe}"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Log execution details
            logger.info(f"Webscraper exit code: {result.returncode}")
            if result.stdout:
                logger.info(f"Webscraper stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Webscraper stderr: {result.stderr}")
            
            # Check for output files
            output_files = list(webscraper_build_dir.glob("*.raw"))
            data_collected = len(output_files) > 0
            
            if data_collected:
                logger.info(f"Data collection successful. Files created: {[f.name for f in output_files]}")
            else:
                logger.warning("No .raw files found after webscraper execution")
            
            return {
                "success": result.returncode == 0,
                "message": f"Webscraper completed with exit code {result.returncode}",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "data_collected": data_collected,
                "output_files": [f.name for f in output_files]
            }
            
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
            logger.debug(f"Restored working directory to: {original_cwd}")
            
    except subprocess.TimeoutExpired:
        error_msg = "Webscraper execution timed out after 5 minutes"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "exit_code": -2,
            "data_collected": False
        }
        
    except Exception as e:
        error_msg = f"Unexpected error during web scraping: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "exit_code": -3,
            "data_collected": False,
            "error_type": type(e).__name__
        }


def analyze_sentiment() -> Dict[str, Union[bool, str, List[Dict], Dict]]:
    """
    Analyze sentiment of scraped data from youtube.raw file.
    
    This function reads the raw YouTube transcript data, processes it through
    the tokenization pipeline, and performs sentiment analysis using the BERT model.
    
    Returns:
        Dict: Analysis results containing sentiment predictions and statistics
    """
    logger.info("Starting sentiment analysis")
    
    # Define paths
    current_dir = Path(__file__).parent
    youtube_raw_path = current_dir / "youtube.raw"
    
    try:
        # Check if youtube.raw exists
        if not youtube_raw_path.exists():
            error_msg = f"YouTube raw data file not found: {youtube_raw_path}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "predictions": [],
                "statistics": {}
            }
        
        logger.info(f"Found raw data file: {youtube_raw_path}")
        
        # Read raw data
        with open(youtube_raw_path, 'r', encoding='utf-8') as f:
            raw_data = f.read()
        
        if not raw_data.strip():
            error_msg = "YouTube raw data file is empty"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "predictions": [],
                "statistics": {}
            }
        
        logger.info(f"Loaded {len(raw_data)} characters of raw data")
        
        # Initialize tokenization pipeline
        tokenization_config = TokenizationConfig()
        tokenization_config.raw_data_path = str(youtube_raw_path)
        
        pipeline = TokenizationPipeline(tokenization_config)
        
        # Preprocess the data
        text_segments = pipeline.preprocess_data(raw_data)
        
        if not text_segments:
            error_msg = "No valid text segments found after preprocessing"
            logger.warning(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "predictions": [],
                "statistics": {"segments_processed": 0}
            }
        
        logger.info(f"Preprocessed into {len(text_segments)} text segments")
        
        # Initialize model configuration and predictor
        model_config = ModelConfig()
        
        # Check if trained model exists
        model_path = Path(model_config.model_save_path) / "best_model.pt"
        
        if model_path.exists():
            # Use trained model for prediction
            logger.info("Using trained model for sentiment prediction")
            predictor = SentimentPredictor(str(model_path), model_config)
            
            # Predict sentiment for each segment
            predictions = []
            for i, segment in enumerate(text_segments):
                try:
                    prediction = predictor.predict_text(segment)
                    prediction['segment_id'] = i
                    predictions.append(prediction)
                    
                    if (i + 1) % 10 == 0:
                        logger.debug(f"Processed {i + 1}/{len(text_segments)} segments")
                        
                except Exception as e:
                    logger.warning(f"Error processing segment {i}: {e}")
                    continue
            
        else:
            # Model not trained yet, create mock predictions for demonstration
            logger.warning(f"Trained model not found at {model_path}, creating mock predictions")
            
            import random
            random.seed(42)  # For reproducible results
            
            sentiment_labels = ['negative', 'neutral', 'positive']
            predictions = []
            
            for i, segment in enumerate(text_segments):
                # Create mock prediction
                sentiment_idx = random.randint(0, 2)
                sentiment = sentiment_labels[sentiment_idx]
                confidence = random.uniform(0.4, 0.9)
                
                prediction = {
                    'segment_id': i,
                    'text': segment[:100] + "..." if len(segment) > 100 else segment,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'probabilities': {
                        'negative': random.uniform(0.1, 0.8) if sentiment != 'negative' else confidence,
                        'neutral': random.uniform(0.1, 0.8) if sentiment != 'neutral' else confidence,
                        'positive': random.uniform(0.1, 0.8) if sentiment != 'positive' else confidence
                    }
                }
                predictions.append(prediction)
        
        # Generate statistics
        sentiment_counts = {}
        confidence_scores = []
        
        for pred in predictions:
            sentiment = pred['sentiment']
            confidence = pred['confidence']
            
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            confidence_scores.append(confidence)
        
        # Calculate statistics
        statistics = {
            "total_segments": len(text_segments),
            "predictions_made": len(predictions),
            "sentiment_distribution": sentiment_counts,
            "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "min_confidence": min(confidence_scores) if confidence_scores else 0,
            "max_confidence": max(confidence_scores) if confidence_scores else 0,
            "data_size_chars": len(raw_data),
            "model_used": "trained_bert" if model_path.exists() else "mock_predictions"
        }
        
        # Calculate overall sentiment
        if sentiment_counts:
            overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            statistics["overall_sentiment"] = overall_sentiment
            statistics["sentiment_confidence"] = sentiment_counts[overall_sentiment] / len(predictions)
        
        logger.info(f"Sentiment analysis completed successfully")
        logger.info(f"Processed {len(predictions)} segments")
        logger.info(f"Overall sentiment: {statistics.get('overall_sentiment', 'unknown')}")
        logger.info(f"Average confidence: {statistics['average_confidence']:.3f}")
        
        return {
            "success": True,
            "message": f"Sentiment analysis completed for {len(predictions)} segments",
            "predictions": predictions,
            "statistics": statistics,
            "raw_data_path": str(youtube_raw_path)
        }
        
    except Exception as e:
        error_msg = f"Error during sentiment analysis: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "predictions": [],
            "statistics": {},
            "error_type": type(e).__name__
        }


def full_pipeline() -> Dict[str, Union[bool, str, Dict]]:
    """
    Execute the complete pipeline: web scraping followed by sentiment analysis.
    
    Returns:
        Dict: Combined results from both web scraping and sentiment analysis
    """
    logger.info("=== Starting Full Sentiment Analysis Pipeline ===")
    
    # Step 1: Pull data from web
    scraping_result = pull_from_web()
    
    if not scraping_result["success"]:
        logger.error("Web scraping failed, aborting pipeline")
        return {
            "success": False,
            "message": "Pipeline failed at web scraping step",
            "scraping_result": scraping_result,
            "analysis_result": None
        }
    
    logger.info("Web scraping completed successfully, proceeding to sentiment analysis")
    
    # Step 2: Analyze sentiment
    analysis_result = analyze_sentiment()
    
    # Combine results
    pipeline_success = scraping_result["success"] and analysis_result["success"]
    
    result = {
        "success": pipeline_success,
        "message": "Full pipeline completed" if pipeline_success else "Pipeline completed with errors",
        "scraping_result": scraping_result,
        "analysis_result": analysis_result
    }
    
    if pipeline_success:
        logger.info("=== Full Pipeline Completed Successfully ===")
        if analysis_result.get("statistics"):
            stats = analysis_result["statistics"]
            logger.info(f"Final Results: {stats.get('predictions_made', 0)} predictions, "
                       f"Overall sentiment: {stats.get('overall_sentiment', 'unknown')}")
    else:
        logger.error("=== Pipeline Completed with Errors ===")
    
    return result


def get_latest_analysis() -> Dict[str, Union[bool, str, Dict]]:
    """
    Get the most recent sentiment analysis results without re-running analysis.
    
    This function looks for existing processed data and returns the latest results.
    
    Returns:
        Dict: Latest analysis results if available
    """
    logger.info("Retrieving latest sentiment analysis results")
    
    current_dir = Path(__file__).parent
    
    # Check for processed data files
    processed_data_dir = current_dir / "processed_data"
    stats_file = processed_data_dir / "data_statistics.json"
    
    try:
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                statistics = json.load(f)
            
            logger.info("Found existing analysis statistics")
            return {
                "success": True,
                "message": "Retrieved latest analysis results",
                "statistics": statistics,
                "source": "processed_data_cache"
            }
        else:
            return {
                "success": False,
                "message": "No previous analysis results found. Run analyze_sentiment() first.",
                "statistics": {}
            }
            
    except Exception as e:
        error_msg = f"Error retrieving latest analysis: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "statistics": {},
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    # Demo usage
    print("=== Sentiment Model Route Demo ===")
    
    # Test web scraping
    print("\n1. Testing web scraper...")
    scrape_result = pull_from_web()
    print(f"Scraping result: {scrape_result['success']}")
    print(f"Message: {scrape_result['message']}")
    
    if scrape_result['success']:
        # Test sentiment analysis
        print("\n2. Testing sentiment analysis...")
        analysis_result = analyze_sentiment()
        print(f"Analysis result: {analysis_result['success']}")
        print(f"Message: {analysis_result['message']}")
        
        if analysis_result['success']:
            stats = analysis_result['statistics']
            print(f"Segments processed: {stats.get('predictions_made', 0)}")
            print(f"Overall sentiment: {stats.get('overall_sentiment', 'unknown')}")
    
    print("\n=== Demo Complete ===")