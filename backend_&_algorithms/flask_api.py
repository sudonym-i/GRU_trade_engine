#!/usr/bin/env python3
"""
Flask API Wrapper for Neural Trade Engine

This Flask application provides HTTP endpoints to control and access the 
backend_&_algorithms functionality for stock prediction and analysis.

Usage:
    python flask_api.py
    
Endpoints:
    GET  /api/status              - Get API status
    POST /api/stock/select        - Select stock to follow
    POST /api/model/train         - Train model for selected stock
    GET  /api/model/predict       - Get latest prediction
    POST /api/webscrape           - Trigger webscraping
    GET  /api/predictions/history - Get prediction history
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Add current directory to path for engine imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from engine import (
        train_model, 
        predict_price, 
        get_model_info, 
        list_available_models,
        analyze_sentiment,
        pull_from_web
    )
except ImportError as e:
    print(f"❌ Failed to import engine functions: {e}")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Global state - in production, use a database
app_state = {
    "selected_stock": None,
    "model_trained": False,
    "model_path": None,
    "last_prediction": None,
    "last_webscrape": None,
    "predictions_history": [],
    "portfolio": {
        "cash": 10000.0,  # Starting cash
        "shares": 0,
        "stock_symbol": None
    }
}

# Data directory for storing predictions and logs
DATA_DIR = Path("api_data")
DATA_DIR.mkdir(exist_ok=True)

def save_state():
    """Save current app state to file"""
    with open(DATA_DIR / "app_state.json", "w") as f:
        json.dump(app_state, f, indent=2, default=str)

def load_state():
    """Load app state from file"""
    global app_state
    state_file = DATA_DIR / "app_state.json"
    if state_file.exists():
        with open(state_file, "r") as f:
            loaded_state = json.load(f)
            app_state.update(loaded_state)

# Load state on startup
load_state()

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current API status and configuration"""
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "selected_stock": app_state["selected_stock"],
        "model_trained": app_state["model_trained"],
        "model_path": app_state["model_path"],
        "last_prediction": app_state["last_prediction"],
        "last_webscrape": app_state["last_webscrape"],
        "portfolio": app_state["portfolio"],
        "scheduler_jobs": len(scheduler.get_jobs())
    })

@app.route('/api/stock/select', methods=['POST'])
def select_stock():
    """Select a stock to follow and analyze"""
    data = request.get_json()
    
    if not data or 'ticker' not in data:
        return jsonify({"error": "Missing 'ticker' in request body"}), 400
    
    ticker = data['ticker'].upper()
    
    # Validate ticker format (basic validation)
    if not ticker.isalpha() or len(ticker) > 5:
        return jsonify({"error": "Invalid ticker format"}), 400
    
    # Update state
    app_state["selected_stock"] = ticker
    app_state["model_trained"] = False
    app_state["model_path"] = None
    app_state["portfolio"]["stock_symbol"] = ticker
    
    save_state()
    
    return jsonify({
        "success": True,
        "selected_stock": ticker,
        "message": f"Selected {ticker} for tracking"
    })

@app.route('/api/model/train', methods=['POST'])
def train_model_endpoint():
    """Train model for the selected stock"""
    if not app_state["selected_stock"]:
        return jsonify({"error": "No stock selected. Use /api/stock/select first"}), 400
    
    data = request.get_json() or {}
    days = data.get('days', 730)  # Default to 2 years
    
    try:
        ticker = app_state["selected_stock"]
        
        # Create model directory
        model_dir = Path("models") / ticker
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Train the model
        trainer = train_model([ticker], days=days)
        
        # Save model path
        model_path = f"models/{ticker}/latest_model.pth"
        app_state["model_path"] = model_path
        app_state["model_trained"] = True
        
        save_state()
        
        return jsonify({
            "success": True,
            "ticker": ticker,
            "model_path": model_path,
            "training_days": days,
            "message": f"Model trained successfully for {ticker}"
        })
        
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

@app.route('/api/model/predict', methods=['GET'])
def predict_endpoint():
    """Get prediction for the selected stock"""
    if not app_state["selected_stock"] or not app_state["model_trained"]:
        return jsonify({"error": "No trained model available"}), 400
    
    try:
        ticker = app_state["selected_stock"]
        model_path = app_state["model_path"]
        
        # Make prediction
        prediction = predict_price(ticker, model_path=model_path)
        
        # Update state
        prediction_data = {
            "ticker": ticker,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path
        }
        
        app_state["last_prediction"] = prediction_data
        app_state["predictions_history"].append(prediction_data)
        
        # Keep only last 100 predictions
        if len(app_state["predictions_history"]) > 100:
            app_state["predictions_history"] = app_state["predictions_history"][-100:]
        
        save_state()
        
        return jsonify({
            "success": True,
            **prediction_data
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/api/webscrape', methods=['POST'])
def webscrape_endpoint():
    """Trigger webscraping for sentiment analysis"""
    if not app_state["selected_stock"]:
        return jsonify({"error": "No stock selected"}), 400
    
    try:
        ticker = app_state["selected_stock"]
        output_name = f"{ticker}_sentiment_{datetime.now().strftime('%Y%m%d')}"
        
        # Perform webscraping
        pull_from_web(ticker, output_name)
        
        # Update state
        app_state["last_webscrape"] = {
            "ticker": ticker,
            "output_name": output_name,
            "timestamp": datetime.now().isoformat()
        }
        
        save_state()
        
        return jsonify({
            "success": True,
            "ticker": ticker,
            "output_name": output_name,
            "timestamp": app_state["last_webscrape"]["timestamp"]
        })
        
    except Exception as e:
        return jsonify({"error": f"Webscraping failed: {str(e)}"}), 500

@app.route('/api/predictions/history', methods=['GET'])
def get_predictions_history():
    """Get historical predictions"""
    limit = request.args.get('limit', 50, type=int)
    
    history = app_state["predictions_history"][-limit:]
    
    return jsonify({
        "success": True,
        "count": len(history),
        "predictions": history
    })

@app.route('/api/sentiment/analyze', methods=['POST'])
def analyze_sentiment_endpoint():
    """Analyze sentiment of provided text"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400
    
    try:
        text = data['text']
        sentiment_result = analyze_sentiment(text)
        
        return jsonify({
            "success": True,
            "text": text,
            **sentiment_result
        })
        
    except Exception as e:
        return jsonify({"error": f"Sentiment analysis failed: {str(e)}"}), 500

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Get current portfolio status"""
    return jsonify({
        "success": True,
        "portfolio": app_state["portfolio"]
    })

@app.route('/api/schedule/start', methods=['POST'])
def start_scheduled_jobs():
    """Start scheduled webscraping and predictions"""
    if not app_state["selected_stock"] or not app_state["model_trained"]:
        return jsonify({"error": "Need selected stock and trained model"}), 400
    
    # Remove existing jobs
    scheduler.remove_all_jobs()
    
    # Schedule webscraping (daily at 9 AM)
    scheduler.add_job(
        func=scheduled_webscrape,
        trigger=CronTrigger(hour=9, minute=0),
        id='webscrape_job',
        name='Daily Webscraping'
    )
    
    # Schedule predictions (every 2 hours)
    scheduler.add_job(
        func=scheduled_prediction,
        trigger=IntervalTrigger(hours=2),
        id='prediction_job',
        name='2-Hour Predictions'
    )
    
    return jsonify({
        "success": True,
        "message": "Scheduled jobs started",
        "jobs": [
            {"id": "webscrape_job", "schedule": "Daily at 9:00 AM"},
            {"id": "prediction_job", "schedule": "Every 2 hours"}
        ]
    })

@app.route('/api/schedule/stop', methods=['POST'])
def stop_scheduled_jobs():
    """Stop all scheduled jobs"""
    scheduler.remove_all_jobs()
    
    return jsonify({
        "success": True,
        "message": "All scheduled jobs stopped"
    })

def scheduled_webscrape():
    """Background job for webscraping"""
    if app_state["selected_stock"]:
        try:
            ticker = app_state["selected_stock"]
            output_name = f"{ticker}_sentiment_{datetime.now().strftime('%Y%m%d')}"
            pull_from_web(ticker, output_name)
            
            app_state["last_webscrape"] = {
                "ticker": ticker,
                "output_name": output_name,
                "timestamp": datetime.now().isoformat()
            }
            save_state()
            print(f"✅ Scheduled webscrape completed for {ticker}")
            
        except Exception as e:
            print(f"❌ Scheduled webscrape failed: {e}")

def scheduled_prediction():
    """Background job for predictions"""
    if app_state["selected_stock"] and app_state["model_trained"]:
        try:
            ticker = app_state["selected_stock"]
            model_path = app_state["model_path"]
            
            prediction = predict_price(ticker, model_path=model_path)
            
            prediction_data = {
                "ticker": ticker,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat(),
                "model_path": model_path,
                "scheduled": True
            }
            
            app_state["last_prediction"] = prediction_data
            app_state["predictions_history"].append(prediction_data)
            
            if len(app_state["predictions_history"]) > 100:
                app_state["predictions_history"] = app_state["predictions_history"][-100:]
                
            save_state()
            print(f"✅ Scheduled prediction completed for {ticker}: {prediction}")
            
        except Exception as e:
            print(f"❌ Scheduled prediction failed: {e}")

if __name__ == '__main__':
    print("🚀 Starting Flask API for Neural Trade Engine")
    print(f"📊 Selected stock: {app_state.get('selected_stock', 'None')}")
    print(f"🤖 Model trained: {app_state.get('model_trained', False)}")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )