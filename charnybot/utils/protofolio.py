import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum
import pickle


class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradeOrder:
    """Represents a trading order"""
    ticker: str
    order_type: OrderType
    quantity: float
    price: float
    timestamp: datetime
    order_id: str = None

    def __post_init__(self):
        if self.order_id is None:
            self.order_id = f"{self.ticker}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{self.order_type.value}"


@dataclass
class Position:
    """Represents a position in a stock"""
    ticker: str
    quantity: float
    avg_price: float
    current_price: float
    last_updated: datetime

    @property
    def market_value(self) -> float:
        """Calculate current market value of the position"""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss"""
        return self.quantity * (self.current_price - self.avg_price)

    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized profit/loss as percentage"""
        if self.avg_price == 0:
            return 0.0
        return ((self.current_price - self.avg_price) / self.avg_price) * 100


@dataclass
class PortfolioSnapshot:
    """Represents a snapshot of portfolio state at a specific time"""
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, Position]
    reference_index_value: float
    reference_index_name: str = "S&P 500"


class Portfolio:
    """
    A comprehensive portfolio class for trading that manages multiple stocks,
    tracks portions, maintains cash, and handles buy/sell operations.
    """

    def __init__(self,
                 initial_cash: float = 100000.0,
                 reference_index: str = "S&P 500",
                 portfolio_name: str = "Trading Portfolio"):
        """
        Initialize the portfolio

        Args:
            initial_cash: Starting cash amount
            reference_index: Name of the reference index (default: S&P 500)
            portfolio_name: Name of the portfolio
        """
        self.portfolio_name = portfolio_name
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.reference_index = reference_index
        self.current_date = datetime.now()

        # Portfolio holdings
        self.positions: Dict[str, Position] = {}

        # History tracking
        self.trade_history: List[TradeOrder] = []
        self.portfolio_history: List[PortfolioSnapshot] = []

        # Reference index tracking
        self.reference_index_value = 100.0  # Default starting value
        self.reference_index_history: List[Tuple[datetime, float]] = []

        # Performance tracking
        self.total_invested = 0.0
        self.total_realized_pnl = 0.0

    def is_in(self , ticker: str) -> bool:
        """
        :param ticker: check if ticker is in the portfolio
        :return:
        """
        return (not ticker in self.positions.keys()) or self.positions[ticker].quantity == 0

    def get_total_value(self) -> float:
        """Calculate total portfolio value (cash + all positions)"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    def get_portfolio_weights(self) -> Dict[str, float]:
        """Get the weight/portion of each stock in the portfolio"""
        total_value = self.get_total_value()
        if total_value == 0:
            return {}

        weights = {}
        for ticker, position in self.positions.items():
            weights[ticker] = position.market_value / total_value

        # Add cash weight
        weights['CASH'] = self.cash / total_value

        return weights

    def get_position_summary(self) -> Dict[str, Dict]:
        """Get detailed summary of all positions"""
        summary = {}
        for ticker, position in self.positions.items():
            summary[ticker] = {
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_percent': position.unrealized_pnl_percent,
                'weight': position.market_value / self.get_total_value() if self.get_total_value() > 0 else 0
            }
        return summary

    def update_prices(self, price_updates: Dict[str, float],
                      reference_index_value: Optional[float] = None,
                      update_date: Optional[datetime] = None):
        """
        Update stock prices and reference index value

        Args:
            price_updates: Dict of ticker -> new price
            reference_index_value: New reference index value
            update_date: Date of the update (defaults to current time)
        """
        if update_date is None:
            update_date = datetime.now()

        self.current_date = update_date

        # Update stock prices
        for ticker, new_price in price_updates.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = new_price
                self.positions[ticker].last_updated = update_date

        # Update reference index
        if reference_index_value is not None:
            self.reference_index_value = reference_index_value
            self.reference_index_history.append((update_date, reference_index_value))

        # Take a snapshot for history
        self._take_snapshot()

    def buy_stock(self, ticker: str, quantity: float, price: float,
                  timestamp: Optional[datetime] = None) -> bool:
        """
        Buy a stock

        Args:
            ticker: Stock ticker
            quantity: Number of shares to buy
            price: Price per share
            timestamp: Timestamp of the trade

        Returns:
            bool: True if successful, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()

        total_cost = quantity * price

        # Check if we have enough cash
        if total_cost > self.cash:
            print(f"Insufficient cash. Need ${total_cost:.2f}, have ${self.cash:.2f}")
            return False

        # Create trade order
        order = TradeOrder(
            ticker=ticker,
            order_type=OrderType.BUY,
            quantity=quantity,
            price=price,
            timestamp=timestamp
        )

        # Execute the trade
        self.cash -= total_cost
        self.total_invested += total_cost

        # Update or create position
        if ticker in self.positions:
            # Update existing position
            pos = self.positions[ticker]
            total_quantity = pos.quantity + quantity
            total_cost_basis = (pos.quantity * pos.avg_price) + total_cost
            new_avg_price = total_cost_basis / total_quantity

            pos.quantity = total_quantity
            pos.avg_price = new_avg_price
            pos.current_price = price
            pos.last_updated = timestamp
        else:
            # Create new position
            self.positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                last_updated=timestamp
            )

        # Record the trade
        self.trade_history.append(order)

        print(f"Bought {quantity} shares of {ticker} at ${price:.2f} per share")
        return True

    def sell_stock(self, ticker: str, quantity: float, price: float,
                   timestamp: Optional[datetime] = None) -> bool:
        """
        Sell a stock

        Args:
            ticker: Stock ticker
            quantity: Number of shares to sell
            price: Price per share
            timestamp: Timestamp of the trade

        Returns:
            bool: True if successful, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Check if we have enough shares
        if ticker not in self.positions or self.positions[ticker].quantity < quantity:
            print(
                f"Insufficient shares. Need {quantity} shares of {ticker}, have {self.positions[ticker].quantity if ticker in self.positions else 0}")
            return False

        # Create trade order
        order = TradeOrder(
            ticker=ticker,
            order_type=OrderType.SELL,
            quantity=quantity,
            price=price,
            timestamp=timestamp
        )

        # Execute the trade
        total_proceeds = quantity * price
        self.cash += total_proceeds

        # Calculate realized P&L
        position = self.positions[ticker]
        realized_pnl = quantity * (price - position.avg_price)
        self.total_realized_pnl += realized_pnl

        # Update position
        position.quantity -= quantity
        position.current_price = price
        position.last_updated = timestamp

        # Remove position if quantity becomes zero
        if position.quantity <= 0:
            del self.positions[ticker]

        # Record the trade
        self.trade_history.append(order)

        print(f"Sold {quantity} shares of {ticker} at ${price:.2f} per share. Realized P&L: ${realized_pnl:.2f}")
        return True

    def _take_snapshot(self):
        """Take a snapshot of current portfolio state"""
        snapshot = PortfolioSnapshot(
            timestamp=self.current_date,
            total_value=self.get_total_value(),
            cash=self.cash,
            positions=self.positions.copy(),
            reference_index_value=self.reference_index_value,
            reference_index_name=self.reference_index
        )
        self.portfolio_history.append(snapshot)

    def get_performance_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        if not self.portfolio_history:
            return {}

        initial_value = self.initial_cash
        current_value = self.get_total_value()
        total_return = current_value - initial_value
        total_return_percent = (total_return / initial_value) * 100 if initial_value > 0 else 0

        # Calculate reference index performance
        if self.reference_index_history:
            initial_ref = self.reference_index_history[0][1]
            current_ref = self.reference_index_history[-1][1]
            ref_return_percent = ((current_ref - initial_ref) / initial_ref) * 100 if initial_ref > 0 else 0
        else:
            ref_return_percent = 0

        return {
            'total_value': current_value,
            'total_return': total_return,
            'total_return_percent': total_return_percent,
            'cash': self.cash,
            'invested_value': current_value - self.cash,
            'realized_pnl': self.total_realized_pnl,
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
            'reference_index_return_percent': ref_return_percent,
            'excess_return': total_return_percent - ref_return_percent,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history)
        }

    def save_history(self, filepath: str, format: str = 'pickle'):
        """
        Save portfolio history to file

        Args:
            filepath: Path to save the file
            format: 'pickle' or 'json'
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if format.lower() == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'portfolio_history': self.portfolio_history,
                    'trade_history': self.trade_history,
                    'reference_index_history': self.reference_index_history,
                    'portfolio_name': self.portfolio_name,
                    'initial_cash': self.initial_cash
                }, f)
        elif format.lower() == 'json':
            # Convert to JSON-serializable format
            data = {
                'portfolio_history': [
                    {
                        'timestamp': snap.timestamp.isoformat(),
                        'total_value': snap.total_value,
                        'cash': snap.cash,
                        'positions': {
                            ticker: {
                                'quantity': pos.quantity,
                                'avg_price': pos.avg_price,
                                'current_price': pos.current_price,
                                'last_updated': pos.last_updated.isoformat()
                            } for ticker, pos in snap.positions.items()
                        },
                        'reference_index_value': snap.reference_index_value,
                        'reference_index_name': snap.reference_index_name
                    } for snap in self.portfolio_history
                ],
                'trade_history': [
                    {
                        'ticker': trade.ticker,
                        'order_type': trade.order_type.value,
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'timestamp': trade.timestamp.isoformat(),
                        'order_id': trade.order_id
                    } for trade in self.trade_history
                ],
                'reference_index_history': [
                    {'timestamp': ts.isoformat(), 'value': val}
                    for ts, val in self.reference_index_history
                ],
                'portfolio_name': self.portfolio_name,
                'initial_cash': self.initial_cash
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        print(f"Portfolio history saved to {filepath}")

    def load_history(self, filepath: str, format: str = 'pickle'):
        """
        Load portfolio history from file

        Args:
            filepath: Path to the history file
            format: 'pickle' or 'json'
        """
        if not os.path.exists(filepath):
            print(f"History file {filepath} not found")
            return

        if format.lower() == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        elif format.lower() == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Convert JSON data back to objects
            self.portfolio_history = []
            for snap_data in data['portfolio_history']:
                positions = {}
                for ticker, pos_data in snap_data['positions'].items():
                    positions[ticker] = Position(
                        ticker=ticker,
                        quantity=pos_data['quantity'],
                        avg_price=pos_data['avg_price'],
                        current_price=pos_data['current_price'],
                        last_updated=datetime.fromisoformat(pos_data['last_updated'])
                    )

                snapshot = PortfolioSnapshot(
                    timestamp=datetime.fromisoformat(snap_data['timestamp']),
                    total_value=snap_data['total_value'],
                    cash=snap_data['cash'],
                    positions=positions,
                    reference_index_value=snap_data['reference_index_value'],
                    reference_index_name=snap_data['reference_index_name']
                )
                self.portfolio_history.append(snapshot)

            self.trade_history = []
            for trade_data in data['trade_history']:
                trade = TradeOrder(
                    ticker=trade_data['ticker'],
                    order_type=OrderType(trade_data['order_type']),
                    quantity=trade_data['quantity'],
                    price=trade_data['price'],
                    timestamp=datetime.fromisoformat(trade_data['timestamp']),
                    order_id=trade_data['order_id']
                )
                self.trade_history.append(trade)

            self.reference_index_history = [
                (datetime.fromisoformat(item['timestamp']), item['value'])
                for item in data['reference_index_history']
            ]

            self.portfolio_name = data['portfolio_name']
            self.initial_cash = data['initial_cash']

            # Restore current state from latest snapshot
            if self.portfolio_history:
                latest = self.portfolio_history[-1]
                self.cash = latest.cash
                self.positions = latest.positions.copy()
                self.current_date = latest.timestamp
                self.reference_index_value = latest.reference_index_value

        print(f"Portfolio history loaded from {filepath}")

    def get_portfolio_summary(self) -> str:
        """Get a formatted summary of the portfolio"""
        metrics = self.get_performance_metrics()
        weights = self.get_portfolio_weights()

        summary = f"""
=== {self.portfolio_name} Portfolio Summary ===
Date: {self.current_date.strftime('%Y-%m-%d %H:%M:%S')}

Total Value: ${metrics.get('total_value', 0):,.2f}
Cash: ${metrics.get('cash', 0):,.2f}
Invested Value: ${metrics.get('invested_value', 0):,.2f}

Performance:
- Total Return: ${metrics.get('total_return', 0):,.2f} ({metrics.get('total_return_percent', 0):.2f}%)
- Realized P&L: ${metrics.get('realized_pnl', 0):,.2f}
- Unrealized P&L: ${metrics.get('unrealized_pnl', 0):,.2f}
- Reference Index Return: {metrics.get('reference_index_return_percent', 0):.2f}%
- Excess Return: {metrics.get('excess_return', 0):.2f}%

Positions ({metrics.get('num_positions', 0)} stocks):
"""

        for ticker, weight in weights.items():
            if ticker != 'CASH':
                position = self.positions[ticker]
                summary += f"- {ticker}: {weight:.1%} (${position.market_value:,.2f})\n"

        summary += f"Cash: {weights.get('CASH', 0):.1%}\n"
        summary += f"\nTotal Trades: {metrics.get('num_trades', 0)}"

        return summary

    def plot_portfolio_history(self, save_path: Optional[str] = None):
        """Plot portfolio performance over time"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            print("matplotlib not available for plotting")
            return

        if not self.portfolio_history:
            print("No portfolio history to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Portfolio value over time
        dates = [snap.timestamp for snap in self.portfolio_history]
        values = [snap.total_value for snap in self.portfolio_history]
        cash_values = [snap.cash for snap in self.portfolio_history]

        ax1.plot(dates, values, label='Total Portfolio Value', linewidth=2)
        ax1.plot(dates, cash_values, label='Cash', alpha=0.7)
        ax1.axhline(y=self.initial_cash, color='red', linestyle='--', alpha=0.5, label='Initial Investment')
        ax1.set_title(f'{self.portfolio_name} - Portfolio Value Over Time')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Reference index comparison
        if self.reference_index_history:
            ref_dates = [ts for ts, _ in self.reference_index_history]
            ref_values = [val for _, val in self.reference_index_history]

            # Normalize reference index to same scale as portfolio
            ref_normalized = [val / ref_values[0] * self.initial_cash for val in ref_values]

            ax2.plot(ref_dates, ref_normalized, label=f'{self.reference_index}', alpha=0.7)
            ax2.plot(dates, values, label='Portfolio', linewidth=2)
            ax2.set_title('Portfolio vs Reference Index')
            ax2.set_ylabel('Value ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Portfolio chart saved to {save_path}")

        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Create a portfolio
    portfolio = Portfolio(initial_cash=100000, portfolio_name="Test Portfolio")

    # Simulate some trades
    portfolio.buy_stock("AAPL", 100, 150.0)
    portfolio.buy_stock("MSFT", 50, 300.0)
    portfolio.buy_stock("GOOGL", 25, 2800.0)

    # Update prices
    portfolio.update_prices({
        "AAPL": 155.0,
        "MSFT": 310.0,
        "GOOGL": 2850.0
    }, reference_index_value=105.0)

    # Sell some shares
    portfolio.sell_stock("AAPL", 25, 155.0)

    # Print summary
    print(portfolio.get_portfolio_summary())

    # Save history
    portfolio.save_history("test_portfolio_history.pkl")

    # Create new portfolio and load history
    new_portfolio = Portfolio()
    new_portfolio.load_history("test_portfolio_history.pkl")
    print("\nLoaded portfolio summary:")
    print(new_portfolio.get_portfolio_summary())