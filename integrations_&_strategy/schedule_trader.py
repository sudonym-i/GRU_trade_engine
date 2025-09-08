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
import asyncio
from datetime import datetime, time as dt_time
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
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
        self.config = self._load_config()
        self.time_interval = self.config.get('time_interval', '1d')
        
        # Use AsyncIOScheduler for async support
        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_listener(self.job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
    
    def job_listener(self, event):
        """Listen to job events for logging."""
        if event.exception:
            logger.error(f"Job crashed: {event.exception}")
        else:
            logger.info(f"Job executed successfully: {event.job_id}")
    
    def _load_config(self):
        """Load configuration from file."""
        config_path = self.config_path or 'config.json'
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(__file__), config_path)
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {'time_interval': '1d'}
    
    async def run_trading_cycle(self):
        """Execute trading cycle (renamed from run_daily_trading for flexibility)."""
        try:
            logger.info(f"Starting scheduled trading cycle (interval: {self.time_interval})...")
            
            # For daily intervals, check if it's a weekday
            if self.time_interval == '1d':
                today = datetime.now()
                if today.weekday() >= 5:  # Saturday or Sunday
                    logger.info("Market is closed (weekend), skipping trading")
                    return
            
            # For high-frequency intervals, check market hours
            if self.time_interval in ['5min', '15min', '30min', '1hr']:
                if not self._is_market_hours():
                    logger.info(f"Outside market hours, skipping {self.time_interval} trading cycle")
                    return
            
            # Initialize and run trader
            trader = AutomatedTrader(self.config_path)
            await trader.run_daily_cycle()
            
            logger.info("Scheduled trading cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Scheduled trading cycle failed: {e}")
            raise
    
    def setup_dynamic_schedule(self):
        """
        Set up trading schedule based on time_interval from config.
        """
        interval = self.time_interval.lower()
        
        if interval == '1d':
            # Daily trading at market close
            self.scheduler.add_job(
                func=self.run_trading_cycle,
                trigger=CronTrigger(
                    hour=17,
                    minute=0,
                    day_of_week='mon-fri'
                ),
                id='trading_cycle',
                name='Daily Trading Cycle',
                replace_existing=True
            )
            logger.info("Scheduled daily trading for 17:00 (Monday-Friday)")
            
        elif interval == '1hr':
            # Hourly trading during market hours (9:30 AM - 4:00 PM ET)
            self.scheduler.add_job(
                func=self.run_trading_cycle,
                trigger=CronTrigger(
                    minute=0,
                    hour='9-15',  # 9 AM to 3 PM (last hour starts at 3 PM)
                    day_of_week='mon-fri'
                ),
                id='trading_cycle',
                name='Hourly Trading Cycle',
                replace_existing=True
            )
            logger.info("Scheduled hourly trading (9 AM - 4 PM, Monday-Friday)")
            
        elif interval in ['5min', '15min', '30min']:
            # High-frequency trading during market hours
            minutes = int(interval.replace('min', ''))
            
            self.scheduler.add_job(
                func=self.run_trading_cycle,
                trigger=IntervalTrigger(minutes=minutes),
                id='trading_cycle',
                name=f'{interval.title()} Trading Cycle',
                replace_existing=True
            )
            
            # For high-frequency, also add market hours constraint via job check
            logger.info(f"Scheduled {interval} trading (continuous during market hours, Monday-Friday)")
            
        else:
            logger.warning(f"Unknown interval '{interval}', defaulting to daily")
            self.setup_daily_schedule_fallback()
    
    def setup_daily_schedule_fallback(self, run_time: str = "17:00"):
        """
        Fallback to daily trading schedule.
        
        Args:
            run_time: Time to run daily trading (HH:MM format)
        """
        hour, minute = map(int, run_time.split(':'))
        
        self.scheduler.add_job(
            func=self.run_trading_cycle,
            trigger=CronTrigger(
                hour=hour,
                minute=minute,
                day_of_week='mon-fri'
            ),
            id='trading_cycle',
            name='Daily Trading Cycle (Fallback)',
            replace_existing=True
        )
        logger.info(f"Scheduled daily trading for {run_time} (Monday-Friday) - fallback")
    
    def setup_weekly_retraining(self, run_time: str = "20:00"):
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
    
    def _is_market_hours(self):
        """Check if current time is during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)."""
        import pytz
        
        # Get current time in Eastern timezone
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        
        # Check if it's a weekday
        if now_et.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check if it's during market hours (9:30 AM - 4:00 PM ET)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now_et <= market_close
    
    async def start_scheduler(self):
        """Start the scheduler."""
        try:
            logger.info(f"Starting trading scheduler with {self.time_interval} intervals...")
            
            # Set up dynamic schedule based on time interval
            self.setup_dynamic_schedule()
            
            # Set up weekly retraining (only for daily intervals)
            if self.time_interval == '1d':
                self.setup_weekly_retraining("20:00")
            
            # Start scheduler
            logger.info("Scheduler started. Press Ctrl+C to stop.")
            self.scheduler.start()
            
            # Keep the event loop running
            while True:
                await asyncio.sleep(1)
                
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
        asyncio.run(scheduler.start_scheduler())
    
    elif args.status:
        scheduler = TradingScheduler(config_path)
        scheduler.show_jobs()
    
    elif args.stop:
        # Note: This would need a proper daemon setup to work
        print("To stop the scheduler, press Ctrl+C in the running terminal")
    
    elif args.test:
        async def run_test():
            logger.info("Running test trading cycle...")
            trader = AutomatedTrader(config_path)
            target_stock = args.stock.upper() if args.stock else None
            await trader.run_daily_cycle(target_stock)
            logger.info("Test trading cycle completed")
        
        asyncio.run(run_test())


if __name__ == "__main__":
    main()