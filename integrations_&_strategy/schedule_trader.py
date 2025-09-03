#!/usr/bin/env python3
"""
Scheduled Trading Script for Neural Trade Engine

This script sets up automatic scheduling to run the automated trader daily
after market close. Uses APScheduler for reliable job scheduling.

Usage:
    python schedule_trader.py --start
    python schedule_trader.py --status
    python schedule_trader.py --stop
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, time as dt_time
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

# Import the automated trader
from automated_trader import AutomatedTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingScheduler:
    """
    Scheduler for automated trading operations.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the trading scheduler.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.scheduler = BlockingScheduler()
        self.scheduler.add_listener(self.job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
    
    def job_listener(self, event):
        """Listen to job events for logging."""
        if event.exception:
            logger.error(f"Job crashed: {event.exception}")
        else:
            logger.info(f"Job executed successfully: {event.job_id}")
    
    def run_daily_trading(self):
        """Execute daily trading cycle."""
        try:
            logger.info("Starting scheduled daily trading cycle...")
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            today = datetime.now()
            if today.weekday() >= 5:  # Saturday or Sunday
                logger.info("Market is closed (weekend), skipping trading")
                return
            
            # Initialize and run trader
            trader = AutomatedTrader(self.config_path)
            trader.run_daily_cycle()
            
            logger.info("Scheduled daily trading cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Scheduled trading cycle failed: {e}")
            raise
    
    def setup_daily_schedule(self, run_time: str = "17:00"):
        """
        Set up daily trading schedule.
        
        Args:
            run_time: Time to run daily trading (HH:MM format)
        """
        hour, minute = map(int, run_time.split(':'))
        
        # Schedule daily trading (Monday-Friday at specified time)
        self.scheduler.add_job(
            func=self.run_daily_trading,
            trigger=CronTrigger(
                hour=hour,
                minute=minute,
                day_of_week='mon-fri'
            ),
            id='daily_trading',
            name='Daily Trading Cycle',
            replace_existing=True
        )
        
        logger.info(f"Scheduled daily trading for {run_time} (Monday-Friday)")
    
    def setup_weekly_retraining(self, run_time: str = "18:00"):
        """
        Set up weekly model retraining schedule.
        
        Args:
            run_time: Time to run weekly retraining (HH:MM format)
        """
        hour, minute = map(int, run_time.split(':'))
        
        # Schedule weekly retraining (Sundays)
        self.scheduler.add_job(
            func=self.run_weekly_retraining,
            trigger=CronTrigger(
                hour=hour,
                minute=minute,
                day_of_week='sun'
            ),
            id='weekly_retraining',
            name='Weekly Model Retraining',
            replace_existing=True
        )
        
        logger.info(f"Scheduled weekly retraining for Sundays at {run_time}")
    
    def run_weekly_retraining(self):
        """Execute weekly model retraining."""
        try:
            logger.info("Starting weekly model retraining...")
            
            # Load configuration to get target stock
            trader = AutomatedTrader(self.config_path)
            ticker = trader.config.get('target_stock', 'NVDA')
            
            # Run retraining for the target stock
            backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend_&_algorithms')
            
            import subprocess
            logger.info(f"Retraining model for {ticker}...")
            
            cmd = [
                'python3', 'main.py', 'train',
                '--ticker', ticker,
                '--days', '730',
                '--epochs', '50'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=backend_path,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully retrained model for {ticker}")
            else:
                logger.error(f"Retraining failed for {ticker}: {result.stderr}")
            
            logger.info("Weekly model retraining completed")
            
        except Exception as e:
            logger.error(f"Weekly retraining failed: {e}")
            raise
    
    def start_scheduler(self):
        """Start the scheduler."""
        try:
            logger.info("Starting trading scheduler...")
            
            # Set up schedules
            self.setup_daily_schedule("17:00")  # 5 PM daily trading
            self.setup_weekly_retraining("20:00")  # 8 PM Sunday retraining
            
            # Start scheduler
            logger.info("Scheduler started. Press Ctrl+C to stop.")
            self.scheduler.start()
            
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            self.scheduler.shutdown()
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            self.scheduler.shutdown()
            raise
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")
        else:
            logger.info("Scheduler is not running")
    
    def show_jobs(self):
        """Show scheduled jobs."""
        jobs = self.scheduler.get_jobs()
        
        if not jobs:
            print("No scheduled jobs")
            return
        
        print("\nScheduled Jobs:")
        print("-" * 50)
        
        for job in jobs:
            print(f"ID: {job.id}")
            print(f"Name: {job.name}")
            print(f"Next Run: {job.next_run_time}")
            print(f"Trigger: {job.trigger}")
            print("-" * 50)


def main():
    """Main entry point for scheduler."""
    parser = argparse.ArgumentParser(description='Trading Scheduler')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', action='store_true',
                      help='Start the scheduler')
    group.add_argument('--status', action='store_true',
                      help='Show scheduler status')
    group.add_argument('--stop', action='store_true',
                      help='Stop the scheduler')
    group.add_argument('--test', action='store_true',
                      help='Run one trading cycle for testing')
    
    parser.add_argument('--stock', type=str,
                       help='Target stock ticker (overrides config)')
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    if args.start:
        scheduler = TradingScheduler(config_path)
        scheduler.start_scheduler()
    
    elif args.status:
        scheduler = TradingScheduler(config_path)
        scheduler.show_jobs()
    
    elif args.stop:
        # Note: This would need a proper daemon setup to work
        print("To stop the scheduler, press Ctrl+C in the running terminal")
    
    elif args.test:
        logger.info("Running test trading cycle...")
        trader = AutomatedTrader(config_path)
        target_stock = args.stock.upper() if args.stock else None
        trader.run_daily_cycle(target_stock)
        logger.info("Test trading cycle completed")


if __name__ == "__main__":
    main()