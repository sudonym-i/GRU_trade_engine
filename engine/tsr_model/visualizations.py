import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_loss(losses, save_path=None):
    """
    Plot training loss over epochs
    
    Args:
        losses: list of loss values per epoch
        save_path: path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2, marker='o')
    plt.title('Training Loss Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training loss plot saved to {save_path}")
    else:
        # Auto-save with timestamp if no path provided
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_save_path = f"training_loss_{timestamp}.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Training loss plot saved to {auto_save_path}")
        
        # Try to open the image with the default viewer
        try:
            os.system(f"xdg-open {auto_save_path}")
        except:
            print("Could not automatically open the plot. Please open the saved file manually.")
    
    plt.close()  # Close figure to free memory


def plot_portfolio_performance(portfolio_values, trades, initial_balance, save_path=None):
    """
    Plot portfolio value over time with buy/sell markers
    
    Args:
        portfolio_values: list of portfolio values over time
        trades: list of trade dictionaries from TradingSimulator
        initial_balance: initial portfolio value
        save_path: path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # Portfolio value over time
    time_steps = range(len(portfolio_values))
    ax1.plot(time_steps, portfolio_values, 'b-', linewidth=2, label='Portfolio Value')
    ax1.axhline(y=initial_balance, color='r', linestyle='--', alpha=0.7, label='Initial Balance')
    
    # Mark buy and sell points
    buy_points = []
    sell_points = []
    
    for i, trade in enumerate(trades):
        if trade['action'] == 'buy':
            buy_points.append((i, portfolio_values[i] if i < len(portfolio_values) else portfolio_values[-1]))
        elif trade['action'] == 'sell':
            sell_points.append((i, portfolio_values[i] if i < len(portfolio_values) else portfolio_values[-1]))
    
    if buy_points:
        buy_x, buy_y = zip(*buy_points)
        ax1.scatter(buy_x, buy_y, color='green', marker='^', s=100, label='Buy', zorder=5)
    
    if sell_points:
        sell_x, sell_y = zip(*sell_points)
        ax1.scatter(sell_x, sell_y, color='red', marker='v', s=100, label='Sell', zorder=5)
    
    ax1.set_title('Portfolio Performance Over Time', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Profit/Loss per trade
    trade_profits = [trade['profit'] for trade in trades if trade['profit'] != 0]
    if trade_profits:
        trade_numbers = range(1, len(trade_profits) + 1)
        colors = ['green' if p > 0 else 'red' for p in trade_profits]
        ax2.bar(trade_numbers, trade_profits, color=colors, alpha=0.7)
        ax2.set_title('Profit/Loss Per Trade', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Trade Number', fontsize=12)
        ax2.set_ylabel('Profit/Loss ($)', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No completed trades', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Profit/Loss Per Trade', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Portfolio performance plot saved to {save_path}")
    else:
        # Auto-save with timestamp if no path provided
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_save_path = f"portfolio_performance_{timestamp}.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Portfolio performance plot saved to {auto_save_path}")
        
        # Try to open the image with the default viewer
        try:
            os.system(f"xdg-open {auto_save_path}")
        except:
            print("Could not automatically open the plot. Please open the saved file manually.")
    
    plt.close()  # Close figure to free memory


def plot_price_predictions(actual_prices, predicted_prices, dates=None, save_path=None):
    """
    Plot actual vs predicted prices
    
    Args:
        actual_prices: array of actual prices
        predicted_prices: array of predicted prices
        dates: array of dates (optional)
        save_path: path to save the plot (optional)
    """
    plt.figure(figsize=(15, 8))
    
    if dates is None:
        dates = range(len(actual_prices))
    
    plt.plot(dates, actual_prices, 'b-', linewidth=2, label='Actual Price', alpha=0.8)
    plt.plot(dates, predicted_prices, 'r--', linewidth=2, label='Predicted Price', alpha=0.8)
    
    plt.title('Actual vs Predicted Stock Prices', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Price predictions plot saved to {save_path}")
    else:
        # Auto-save with timestamp if no path provided
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_save_path = f"price_predictions_{timestamp}.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Price predictions plot saved to {auto_save_path}")
        
        # Try to open the image with the default viewer
        try:
            os.system(f"xdg-open {auto_save_path}")
        except:
            print("Could not automatically open the plot. Please open the saved file manually.")
    
    plt.close()  # Close figure to free memory


def plot_technical_indicators(data, ticker, save_path=None):
    """
    Plot stock price with technical indicators
    
    Args:
        data: DataFrame with OHLCV data and technical indicators
        ticker: stock ticker symbol
        save_path: path to save the plot (optional)
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[3, 1, 1])
    
    # Price and SMA
    axes[0].plot(data.index, data['Close'], label='Close Price', linewidth=2)
    if 'SMA_14' in data.columns:
        axes[0].plot(data.index, data['SMA_14'], label='SMA (14)', linewidth=1, alpha=0.8)
    axes[0].set_title(f'{ticker} - Price and Moving Average', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price ($)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RSI
    if 'RSI_14' in data.columns:
        axes[1].plot(data.index, data['RSI_14'], color='orange', linewidth=2)
        axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        axes[1].fill_between(data.index, 30, 70, alpha=0.1, color='gray')
        axes[1].set_title('RSI (14)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('RSI', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
    
    # MACD
    if 'MACD' in data.columns:
        axes[2].plot(data.index, data['MACD'], label='MACD', linewidth=2, color='blue')
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2].set_title('MACD', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('MACD', fontsize=12)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Technical indicators plot saved to {save_path}")
    else:
        # Auto-save with timestamp if no path provided
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_save_path = f"technical_indicators_{ticker}_{timestamp}.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Technical indicators plot saved to {auto_save_path}")
        
        # Try to open the image with the default viewer
        try:
            os.system(f"xdg-open {auto_save_path}")
        except:
            print("Could not automatically open the plot. Please open the saved file manually.")
    
    plt.close()  # Close figure to free memory


def create_interactive_dashboard(data, portfolio_values, trades, ticker):
    """
    Create an interactive dashboard using Plotly
    
    Args:
        data: DataFrame with OHLCV data and technical indicators
        portfolio_values: list of portfolio values over time
        trades: list of trade dictionaries
        ticker: stock ticker symbol
    
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f'{ticker} - Stock Price & Technical Indicators',
            'RSI (14)',
            'MACD',
            'Portfolio Value'
        ),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Stock price and SMA
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close Price', 
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    if 'SMA_14' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_14'], name='SMA (14)', 
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    # RSI
    if 'RSI_14' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI_14'], name='RSI', 
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
    
    # MACD
    if 'MACD' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], name='MACD', 
                      line=dict(color='red', width=2)),
            row=3, col=1
        )
        fig.add_hline(y=0, line_color="black", opacity=0.3, row=3, col=1)
    
    # Portfolio value
    portfolio_dates = data.index[-len(portfolio_values):] if len(portfolio_values) <= len(data) else data.index
    fig.add_trace(
        go.Scatter(x=portfolio_dates, y=portfolio_values, name='Portfolio Value', 
                  line=dict(color='green', width=2)),
        row=4, col=1
    )
    
    # Add buy/sell markers
    buy_dates = []
    sell_dates = []
    buy_values = []
    sell_values = []
    
    for i, trade in enumerate(trades):
        if i < len(portfolio_dates):
            if trade['action'] == 'buy':
                buy_dates.append(portfolio_dates[i])
                buy_values.append(portfolio_values[i] if i < len(portfolio_values) else portfolio_values[-1])
            elif trade['action'] == 'sell':
                sell_dates.append(portfolio_dates[i])
                sell_values.append(portfolio_values[i] if i < len(portfolio_values) else portfolio_values[-1])
    
    if buy_dates:
        fig.add_trace(
            go.Scatter(x=buy_dates, y=buy_values, mode='markers', name='Buy',
                      marker=dict(symbol='triangle-up', size=10, color='green')),
            row=4, col=1
        )
    
    if sell_dates:
        fig.add_trace(
            go.Scatter(x=sell_dates, y=sell_values, mode='markers', name='Sell',
                      marker=dict(symbol='triangle-down', size=10, color='red')),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Trading Analysis Dashboard',
        xaxis_title='Date',
        height=1000,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Portfolio ($)", row=4, col=1)
    
    return fig


def plot_performance_metrics(trades, initial_balance, save_path=None):
    """
    Plot various performance metrics
    
    Args:
        trades: list of trade dictionaries
        initial_balance: initial portfolio value
        save_path: path to save the plot (optional)
    """
    # Calculate metrics
    profits = [trade['profit'] for trade in trades if trade['profit'] != 0]
    
    if not profits:
        print("No completed trades to analyze.")
        return
    
    # Create summary statistics
    total_profit = sum(profits)
    win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
    avg_win = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
    avg_loss = np.mean([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 0
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Profit distribution
    axes[0, 0].hist(profits, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_title('Profit/Loss Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Profit/Loss ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative profit
    cumulative_profit = np.cumsum(profits)
    axes[0, 1].plot(range(1, len(cumulative_profit) + 1), cumulative_profit, 
                   'g-', linewidth=2, marker='o')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Cumulative Profit/Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Trade Number')
    axes[0, 1].set_ylabel('Cumulative Profit ($)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Performance metrics table
    axes[1, 0].axis('off')
    metrics_data = [
        ['Total Profit/Loss', f'${total_profit:.2f}'],
        ['Total Return', f'{(total_profit/initial_balance)*100:.1f}%'],
        ['Number of Trades', str(len(profits))],
        ['Win Rate', f'{win_rate:.1f}%'],
        ['Average Win', f'${avg_win:.2f}'],
        ['Average Loss', f'${avg_loss:.2f}'],
        ['Profit Factor', f'{abs(avg_win/avg_loss):.2f}' if avg_loss != 0 else 'N/A']
    ]
    
    table = axes[1, 0].table(cellText=metrics_data,
                            colLabels=['Metric', 'Value'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    axes[1, 0].set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Win/Loss streaks
    win_loss_sequence = ['W' if p > 0 else 'L' for p in profits]
    current_streak = 1
    max_win_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    
    for i in range(len(win_loss_sequence)):
        if i > 0 and win_loss_sequence[i] == win_loss_sequence[i-1]:
            current_streak += 1
        else:
            current_streak = 1
            
        if win_loss_sequence[i] == 'W':
            current_win_streak = current_streak
            max_win_streak = max(max_win_streak, current_win_streak)
            current_loss_streak = 0
        else:
            current_loss_streak = current_streak
            max_loss_streak = max(max_loss_streak, current_loss_streak)
            current_win_streak = 0
    
    streak_data = ['Win Streak', 'Loss Streak']
    streak_values = [max_win_streak, max_loss_streak]
    colors = ['green', 'red']
    
    axes[1, 1].bar(streak_data, streak_values, color=colors, alpha=0.7)
    axes[1, 1].set_title('Maximum Win/Loss Streaks', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Number of Consecutive Trades')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance metrics plot saved to {save_path}")
    else:
        # Auto-save with timestamp if no path provided
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_save_path = f"performance_metrics_{timestamp}.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Performance metrics plot saved to {auto_save_path}")
        
        # Try to open the image with the default viewer
        try:
            os.system(f"xdg-open {auto_save_path}")
        except:
            print("Could not automatically open the plot. Please open the saved file manually.")
    
    plt.close()  # Close figure to free memory