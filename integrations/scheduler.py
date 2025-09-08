#!/usr/bin/env python3
"""
Scheduler Script for Neural Trade Engine

This script runs main.py predict at scheduled intervals based on config.json settings.
It also supports webscraping at longer intervals.
"""

import json
import subprocess
import time
import logging
import os
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path


class TradeEngineScheduler:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_logging()
        self.running = False
        self.predict_thread = None
        self.webscrape_thread = None
        
    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"ERROR: Failed to load config file {self.config_path}: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_file = log_config.get("log_file", "scheduler.log")
        log_level = log_config.get("log_level", "INFO")
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_predict(self):
        """Run the predict command from main.py."""
        try:
            scheduler_config = self.config["scheduler"]
            paths_config = self.config["paths"]
            
            # Build command arguments
            cmd = [
                "python3", paths_config["main_script"],
                "predict",
                "--ticker", scheduler_config["ticker"]
            ]
            
            # Add optional parameters
            if scheduler_config.get("model_path"):
                cmd.extend(["--model", scheduler_config["model_path"]])
            
            if scheduler_config.get("confidence"):
                cmd.append("--confidence")
            
            if scheduler_config.get("interval"):
                cmd.extend(["--interval", scheduler_config["interval"]])
            
            # Change to working directory
            working_dir = paths_config.get("working_directory", ".")
            
            self.logger.info(f"Running prediction: {' '.join(cmd)}")
            
            # Execute command
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info("Prediction completed successfully")
                if result.stdout:
                    self.logger.info(f"Output: {result.stdout.strip()}")
            else:
                self.logger.error(f"Prediction failed with return code {result.returncode}")
                if result.stderr:
                    self.logger.error(f"Error: {result.stderr.strip()}")
                    
        except subprocess.TimeoutExpired:
            self.logger.error("Prediction command timed out")
        except Exception as e:
            self.logger.error(f"Error running prediction: {e}")
    
    def run_webscrape(self):
        """Run the webscrape command from main.py."""
        try:
            scheduler_config = self.config["scheduler"]
            paths_config = self.config["paths"]
            
            # Build command arguments
            cmd = [
                "python3", paths_config["main_script"],
                "webscrape",
                "--ticker", scheduler_config["ticker"]
            ]
            
            # Change to working directory
            working_dir = paths_config.get("working_directory", ".")
            
            self.logger.info(f"Running webscrape: {' '.join(cmd)}")
            
            # Execute command
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info("Webscrape completed successfully")
                if result.stdout:
                    self.logger.info(f"Output: {result.stdout.strip()}")
            else:
                self.logger.error(f"Webscrape failed with return code {result.returncode}")
                if result.stderr:
                    self.logger.error(f"Error: {result.stderr.strip()}")
                    
        except subprocess.TimeoutExpired:
            self.logger.error("Webscrape command timed out")
        except Exception as e:
            self.logger.error(f"Error running webscrape: {e}")
    
    def predict_loop(self):
        """Main loop for running predictions at scheduled intervals."""
        interval_minutes = self.config["scheduler"]["predict_interval_minutes"]
        interval_seconds = interval_minutes * 60
        
        self.logger.info(f"Starting predict loop - running every {interval_minutes} minutes")
        
        while self.running:
            try:
                # Run prediction
                self.run_predict()
                
                # Wait for next interval
                if self.running:  # Check if still running before sleeping
                    self.logger.info(f"Next prediction in {interval_minutes} minutes")
                    time.sleep(interval_seconds)
                    
            except Exception as e:
                self.logger.error(f"Error in predict loop: {e}")
                if self.running:
                    time.sleep(60)  # Wait 1 minute before retrying
    
    def webscrape_loop(self):
        """Main loop for running webscraping at scheduled intervals."""
        interval_hours = self.config["scheduler"]["webscrape_interval_hours"]
        interval_seconds = interval_hours * 3600
        
        self.logger.info(f"Starting webscrape loop - running every {interval_hours} hours")
        
        while self.running:
            try:
                # Run webscrape
                self.run_webscrape()
                
                # Wait for next interval
                if self.running:  # Check if still running before sleeping
                    self.logger.info(f"Next webscrape in {interval_hours} hours")
                    time.sleep(interval_seconds)
                    
            except Exception as e:
                self.logger.error(f"Error in webscrape loop: {e}")
                if self.running:
                    time.sleep(300)  # Wait 5 minutes before retrying
    
    def start(self):
        """Start the scheduler."""
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.logger.info("Starting Neural Trade Engine Scheduler")
        
        # Start predict thread
        self.predict_thread = threading.Thread(target=self.predict_loop)
        self.predict_thread.daemon = True
        self.predict_thread.start()
        
        # Start webscrape thread
        self.webscrape_thread = threading.Thread(target=self.webscrape_loop)
        self.webscrape_thread.daemon = True
        self.webscrape_thread.start()
        
        self.logger.info("Scheduler started successfully")
    
    def stop(self):
        """Stop the scheduler."""
        if not self.running:
            self.logger.warning("Scheduler is not running")
            return
        
        self.logger.info("Stopping scheduler...")
        self.running = False
        
        # Wait for threads to finish (with timeout)
        if self.predict_thread and self.predict_thread.is_alive():
            self.predict_thread.join(timeout=5)
        
        if self.webscrape_thread and self.webscrape_thread.is_alive():
            self.webscrape_thread.join(timeout=5)
        
        self.logger.info("Scheduler stopped")
    
    def run_once(self, command="predict"):
        """Run a single prediction or webscrape command."""
        if command == "predict":
            self.run_predict()
        elif command == "webscrape":
            self.run_webscrape()
        else:
            self.logger.error(f"Unknown command: {command}")


def main():
    """Main entry point for the scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Neural Trade Engine Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', default='config.json',
                       help='Path to configuration file (default: config.json)')
    parser.add_argument('--run-once', choices=['predict', 'webscrape'],
                       help='Run a single command instead of continuous scheduling')
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = TradeEngineScheduler(args.config)
    
    if args.run_once:
        # Run single command
        scheduler.run_once(args.run_once)
    else:
        # Run continuous scheduling
        try:
            scheduler.start()
            
            # Keep main thread alive
            while scheduler.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nShutting down scheduler...")
            scheduler.stop()
        except Exception as e:
            scheduler.logger.error(f"Unexpected error: {e}")
            scheduler.stop()
            sys.exit(1)


if __name__ == "__main__":
    main()