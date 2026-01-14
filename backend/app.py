"""
Trading Strategy Assistant - Backend API
A comprehensive backtesting platform with SQL-based strategy management and reporting.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)
CORS(app)

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "trading.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# =============================================================================
# DATABASE MODELS
# =============================================================================

class Strategy(db.Model):
    """Stores strategy configurations and metadata"""
    __tablename__ = 'strategies'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    strategy_type = db.Column(db.String(50), nullable=False)  # ma_crossover, rsi, macd, bollinger
    parameters = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    backtests = db.relationship('Backtest', backref='strategy', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'strategy_type': self.strategy_type,
            'parameters': self.parameters,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_active': self.is_active
        }


class Backtest(db.Model):
    """Stores backtest execution results"""
    __tablename__ = 'backtests'
    
    id = db.Column(db.Integer, primary_key=True)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategies.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    initial_capital = db.Column(db.Float, default=10000.0)
    
    # Performance metrics
    total_return = db.Column(db.Float)
    annualized_return = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    win_rate = db.Column(db.Float)
    total_trades = db.Column(db.Integer)
    profitable_trades = db.Column(db.Integer)
    avg_trade_return = db.Column(db.Float)
    max_consecutive_wins = db.Column(db.Integer)
    max_consecutive_losses = db.Column(db.Integer)
    profit_factor = db.Column(db.Float)
    
    # Equity curve data (stored as JSON)
    equity_curve = db.Column(db.JSON)
    drawdown_curve = db.Column(db.JSON)
    monthly_returns = db.Column(db.JSON)
    trades = db.Column(db.JSON)
    
    executed_at = db.Column(db.DateTime, default=datetime.utcnow)
    execution_time_ms = db.Column(db.Integer)
    
    def to_dict(self):
        return {
            'id': self.id,
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'avg_trade_return': self.avg_trade_return,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'profit_factor': self.profit_factor,
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve,
            'monthly_returns': self.monthly_returns,
            'trades': self.trades,
            'executed_at': self.executed_at.isoformat(),
            'execution_time_ms': self.execution_time_ms
        }


class PriceData(db.Model):
    """Stores historical price data"""
    __tablename__ = 'price_data'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    date = db.Column(db.Date, nullable=False)
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volume = db.Column(db.BigInteger)
    
    __table_args__ = (db.UniqueConstraint('symbol', 'date', name='unique_symbol_date'),)


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class BacktestEngine:
    """Core backtesting engine with multiple strategy implementations"""
    
    STRATEGY_TEMPLATES = {
        'ma_crossover': {
            'name': 'Moving Average Crossover',
            'description': 'Buy when fast MA crosses above slow MA, sell when it crosses below',
            'parameters': {
                'fast_period': {'type': 'int', 'default': 10, 'min': 2, 'max': 50, 'description': 'Fast MA period'},
                'slow_period': {'type': 'int', 'default': 30, 'min': 10, 'max': 200, 'description': 'Slow MA period'},
                'ma_type': {'type': 'select', 'default': 'sma', 'options': ['sma', 'ema'], 'description': 'Moving average type'}
            }
        },
        'rsi': {
            'name': 'RSI Mean Reversion',
            'description': 'Buy when RSI is oversold, sell when overbought',
            'parameters': {
                'period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50, 'description': 'RSI period'},
                'oversold': {'type': 'int', 'default': 30, 'min': 10, 'max': 40, 'description': 'Oversold threshold'},
                'overbought': {'type': 'int', 'default': 70, 'min': 60, 'max': 90, 'description': 'Overbought threshold'}
            }
        },
        'macd': {
            'name': 'MACD Crossover',
            'description': 'Buy when MACD crosses above signal line, sell when it crosses below',
            'parameters': {
                'fast_period': {'type': 'int', 'default': 12, 'min': 5, 'max': 30, 'description': 'Fast EMA period'},
                'slow_period': {'type': 'int', 'default': 26, 'min': 15, 'max': 50, 'description': 'Slow EMA period'},
                'signal_period': {'type': 'int', 'default': 9, 'min': 5, 'max': 20, 'description': 'Signal line period'}
            }
        },
        'bollinger': {
            'name': 'Bollinger Bands',
            'description': 'Buy at lower band, sell at upper band (mean reversion)',
            'parameters': {
                'period': {'type': 'int', 'default': 20, 'min': 10, 'max': 50, 'description': 'Bollinger period'},
                'std_dev': {'type': 'float', 'default': 2.0, 'min': 1.0, 'max': 3.0, 'description': 'Standard deviation multiplier'}
            }
        },
        'dual_momentum': {
            'name': 'Dual Momentum',
            'description': 'Combines absolute and relative momentum for trend following',
            'parameters': {
                'lookback': {'type': 'int', 'default': 12, 'min': 3, 'max': 24, 'description': 'Momentum lookback (months)'},
                'sma_period': {'type': 'int', 'default': 200, 'min': 50, 'max': 300, 'description': 'Trend filter SMA period'}
            }
        },
        'mean_reversion': {
            'name': 'Mean Reversion',
            'description': 'Buy when price deviates significantly below moving average',
            'parameters': {
                'period': {'type': 'int', 'default': 20, 'min': 5, 'max': 100, 'description': 'Moving average period'},
                'entry_threshold': {'type': 'float', 'default': -2.0, 'min': -5.0, 'max': -0.5, 'description': 'Entry Z-score'},
                'exit_threshold': {'type': 'float', 'default': 0.0, 'min': -1.0, 'max': 1.0, 'description': 'Exit Z-score'}
            }
        }
    }
    
    def __init__(self, prices_df, initial_capital=10000):
        self.prices = prices_df.copy()
        self.initial_capital = initial_capital
        
    def calculate_sma(self, period):
        return self.prices['close'].rolling(window=period).mean()
    
    def calculate_ema(self, period):
        return self.prices['close'].ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, period):
        delta = self.prices['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, fast, slow, signal):
        fast_ema = self.prices['close'].ewm(span=fast, adjust=False).mean()
        slow_ema = self.prices['close'].ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line
    
    def calculate_bollinger(self, period, std_dev):
        sma = self.prices['close'].rolling(window=period).mean()
        std = self.prices['close'].rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, sma, lower
    
    def generate_signals(self, strategy_type, parameters):
        """Generate buy/sell signals based on strategy type"""
        signals = pd.Series(0, index=self.prices.index)
        
        if strategy_type == 'ma_crossover':
            fast_period = parameters.get('fast_period', 10)
            slow_period = parameters.get('slow_period', 30)
            ma_type = parameters.get('ma_type', 'sma')
            
            if ma_type == 'ema':
                fast_ma = self.calculate_ema(fast_period)
                slow_ma = self.calculate_ema(slow_period)
            else:
                fast_ma = self.calculate_sma(fast_period)
                slow_ma = self.calculate_sma(slow_period)
            
            # Generate signals
            signals = pd.Series(0, index=self.prices.index)
            signals[fast_ma > slow_ma] = 1  # Long signal
            signals[fast_ma < slow_ma] = -1  # Exit signal
            
        elif strategy_type == 'rsi':
            period = parameters.get('period', 14)
            oversold = parameters.get('oversold', 30)
            overbought = parameters.get('overbought', 70)
            
            rsi = self.calculate_rsi(period)
            
            signals = pd.Series(0, index=self.prices.index)
            signals[rsi < oversold] = 1  # Buy signal
            signals[rsi > overbought] = -1  # Sell signal
            
        elif strategy_type == 'macd':
            fast = parameters.get('fast_period', 12)
            slow = parameters.get('slow_period', 26)
            signal = parameters.get('signal_period', 9)
            
            macd_line, signal_line = self.calculate_macd(fast, slow, signal)
            
            signals = pd.Series(0, index=self.prices.index)
            signals[macd_line > signal_line] = 1
            signals[macd_line < signal_line] = -1
            
        elif strategy_type == 'bollinger':
            period = parameters.get('period', 20)
            std_dev = parameters.get('std_dev', 2.0)
            
            upper, middle, lower = self.calculate_bollinger(period, std_dev)
            
            signals = pd.Series(0, index=self.prices.index)
            signals[self.prices['close'] < lower] = 1  # Buy at lower band
            signals[self.prices['close'] > upper] = -1  # Sell at upper band
            
        elif strategy_type == 'dual_momentum':
            lookback = parameters.get('lookback', 12) * 21  # Convert months to trading days
            sma_period = parameters.get('sma_period', 200)
            
            momentum = self.prices['close'].pct_change(lookback)
            sma = self.calculate_sma(sma_period)
            
            signals = pd.Series(0, index=self.prices.index)
            signals[(momentum > 0) & (self.prices['close'] > sma)] = 1
            signals[(momentum < 0) | (self.prices['close'] < sma)] = -1
            
        elif strategy_type == 'mean_reversion':
            period = parameters.get('period', 20)
            entry_threshold = parameters.get('entry_threshold', -2.0)
            exit_threshold = parameters.get('exit_threshold', 0.0)
            
            sma = self.calculate_sma(period)
            std = self.prices['close'].rolling(window=period).std()
            z_score = (self.prices['close'] - sma) / std
            
            signals = pd.Series(0, index=self.prices.index)
            signals[z_score < entry_threshold] = 1
            signals[z_score > exit_threshold] = -1
            
        return signals
    
    def run_backtest(self, strategy_type, parameters):
        """Execute backtest and calculate all performance metrics"""
        import time
        start_time = time.time()
        
        signals = self.generate_signals(strategy_type, parameters)
        
        # Initialize tracking variables
        position = 0
        cash = self.initial_capital
        shares = 0
        equity_curve = []
        trades = []
        entry_price = 0
        entry_date = None
        
        # Track drawdown
        peak_equity = self.initial_capital
        drawdown_curve = []
        
        for i, (idx, row) in enumerate(self.prices.iterrows()):
            price = row['close']
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Execute trades
            if signal == 1 and position == 0:  # Buy signal, no position
                shares = cash / price
                cash = 0
                position = 1
                entry_price = price
                entry_date = idx
                
            elif signal == -1 and position == 1:  # Sell signal, have position
                cash = shares * price
                trade_return = (price - entry_price) / entry_price * 100
                trades.append({
                    'entry_date': entry_date.isoformat() if hasattr(entry_date, 'isoformat') else str(entry_date),
                    'exit_date': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(price, 2),
                    'return_pct': round(trade_return, 2),
                    'profit': round(shares * (price - entry_price), 2)
                })
                shares = 0
                position = 0
                entry_price = 0
                entry_date = None
            
            # Calculate equity
            equity = cash + (shares * price)
            equity_curve.append({
                'date': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                'equity': round(equity, 2),
                'price': round(price, 2)
            })
            
            # Track drawdown
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (equity - peak_equity) / peak_equity * 100
            drawdown_curve.append({
                'date': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                'drawdown': round(drawdown, 2)
            })
        
        # Calculate performance metrics
        final_equity = equity_curve[-1]['equity'] if equity_curve else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        # Annualized return
        days = len(self.prices)
        years = days / 252
        annualized_return = ((final_equity / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        equity_values = [e['equity'] for e in equity_curve]
        if len(equity_values) > 1:
            returns = pd.Series(equity_values).pct_change().dropna()
            sharpe = (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        else:
            sharpe = 0
        
        # Max drawdown
        max_drawdown = min([d['drawdown'] for d in drawdown_curve]) if drawdown_curve else 0
        
        # Trade statistics
        total_trades = len(trades)
        profitable_trades = len([t for t in trades if t['return_pct'] > 0])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        avg_trade_return = np.mean([t['return_pct'] for t in trades]) if trades else 0
        
        # Consecutive wins/losses
        max_wins = max_losses = current_wins = current_losses = 0
        for t in trades:
            if t['return_pct'] > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        # Profit factor
        gross_profit = sum([t['profit'] for t in trades if t['profit'] > 0])
        gross_loss = abs(sum([t['profit'] for t in trades if t['profit'] < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Monthly returns
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            equity_df.set_index('date', inplace=True)
            monthly = equity_df['equity'].resample('ME').last()
            monthly_returns = monthly.pct_change().dropna() * 100
            monthly_returns_data = [
                {'month': idx.strftime('%Y-%m'), 'return': round(val, 2)} 
                for idx, val in monthly_returns.items()
            ]
        else:
            monthly_returns_data = []
        
        execution_time = int((time.time() - start_time) * 1000)
        
        return {
            'total_return': round(total_return, 2),
            'annualized_return': round(annualized_return, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_drawdown, 2),
            'win_rate': round(win_rate, 2),
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'avg_trade_return': round(avg_trade_return, 2),
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
            'equity_curve': equity_curve,
            'drawdown_curve': drawdown_curve,
            'monthly_returns': monthly_returns_data,
            'trades': trades,
            'execution_time_ms': execution_time
        }


# =============================================================================
# SAMPLE DATA GENERATION
# =============================================================================

def generate_sample_data(symbol, start_date, end_date, seed=None):
    """Generate realistic stock price data using geometric Brownian motion"""
    if seed:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    n_days = len(dates)
    
    # Parameters for realistic stock behavior
    mu = 0.0008  # Daily drift (~20% annual)
    sigma = 0.02  # Daily volatility (~32% annual)
    initial_price = 100
    
    # Generate returns using geometric Brownian motion
    returns = np.random.normal(mu, sigma, n_days)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    data = []
    for i, date in enumerate(dates):
        close = prices[i]
        daily_range = close * np.random.uniform(0.01, 0.03)
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)
        open_price = close + np.random.uniform(-daily_range/2, daily_range/2)
        volume = int(np.random.uniform(1000000, 10000000))
        
        data.append({
            'symbol': symbol,
            'date': date.date(),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    return data


def seed_database():
    """Initialize database with sample price data for multiple symbols"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'SPY', 'QQQ', 'META']
    
    for i, symbol in enumerate(symbols):
        # Check if data already exists
        existing = PriceData.query.filter_by(symbol=symbol).first()
        if existing:
            continue
            
        # Generate 3 years of data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=3*365)
        
        data = generate_sample_data(symbol, start_date, end_date, seed=42+i)
        
        for row in data:
            price = PriceData(
                symbol=row['symbol'],
                date=row['date'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            db.session.add(price)
        
        db.session.commit()
        print(f"Seeded {len(data)} records for {symbol}")


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})


# Strategy Templates
@app.route('/api/templates', methods=['GET'])
def get_templates():
    """Get all available strategy templates"""
    return jsonify(BacktestEngine.STRATEGY_TEMPLATES)


# Symbols
@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    """Get list of available symbols"""
    symbols = db.session.query(PriceData.symbol).distinct().all()
    return jsonify([s[0] for s in symbols])


# Price Data
@app.route('/api/prices/<symbol>', methods=['GET'])
def get_prices(symbol):
    """Get price data for a symbol"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    query = PriceData.query.filter_by(symbol=symbol.upper())
    
    if start_date:
        query = query.filter(PriceData.date >= datetime.strptime(start_date, '%Y-%m-%d').date())
    if end_date:
        query = query.filter(PriceData.date <= datetime.strptime(end_date, '%Y-%m-%d').date())
    
    prices = query.order_by(PriceData.date).all()
    
    return jsonify([{
        'date': p.date.isoformat(),
        'open': p.open,
        'high': p.high,
        'low': p.low,
        'close': p.close,
        'volume': p.volume
    } for p in prices])


# Strategies CRUD
@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get all strategies with optional filtering"""
    strategy_type = request.args.get('type')
    is_active = request.args.get('active')
    
    query = Strategy.query
    
    if strategy_type:
        query = query.filter_by(strategy_type=strategy_type)
    if is_active is not None:
        query = query.filter_by(is_active=is_active.lower() == 'true')
    
    strategies = query.order_by(Strategy.updated_at.desc()).all()
    return jsonify([s.to_dict() for s in strategies])


@app.route('/api/strategies', methods=['POST'])
def create_strategy():
    """Create a new strategy"""
    data = request.json
    
    strategy = Strategy(
        name=data['name'],
        description=data.get('description', ''),
        strategy_type=data['strategy_type'],
        parameters=data['parameters']
    )
    
    db.session.add(strategy)
    db.session.commit()
    
    return jsonify(strategy.to_dict()), 201


@app.route('/api/strategies/<int:id>', methods=['GET'])
def get_strategy(id):
    """Get a specific strategy"""
    strategy = Strategy.query.get_or_404(id)
    return jsonify(strategy.to_dict())


@app.route('/api/strategies/<int:id>', methods=['PUT'])
def update_strategy(id):
    """Update a strategy"""
    strategy = Strategy.query.get_or_404(id)
    data = request.json
    
    strategy.name = data.get('name', strategy.name)
    strategy.description = data.get('description', strategy.description)
    strategy.parameters = data.get('parameters', strategy.parameters)
    strategy.is_active = data.get('is_active', strategy.is_active)
    
    db.session.commit()
    return jsonify(strategy.to_dict())


@app.route('/api/strategies/<int:id>', methods=['DELETE'])
def delete_strategy(id):
    """Delete a strategy"""
    strategy = Strategy.query.get_or_404(id)
    db.session.delete(strategy)
    db.session.commit()
    return '', 204


# Backtesting
@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Execute a backtest for a strategy"""
    data = request.json
    
    strategy_id = data.get('strategy_id')
    strategy_type = data.get('strategy_type')
    parameters = data.get('parameters')
    symbol = data.get('symbol', 'SPY')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    initial_capital = data.get('initial_capital', 10000)
    
    # If strategy_id provided, load from database
    if strategy_id:
        strategy = Strategy.query.get_or_404(strategy_id)
        strategy_type = strategy.strategy_type
        parameters = strategy.parameters
    
    # Get price data
    query = PriceData.query.filter_by(symbol=symbol.upper())
    if start_date:
        query = query.filter(PriceData.date >= datetime.strptime(start_date, '%Y-%m-%d').date())
    if end_date:
        query = query.filter(PriceData.date <= datetime.strptime(end_date, '%Y-%m-%d').date())
    
    prices = query.order_by(PriceData.date).all()
    
    if not prices:
        return jsonify({'error': 'No price data found for the specified criteria'}), 400
    
    # Convert to DataFrame
    prices_df = pd.DataFrame([{
        'date': p.date,
        'open': p.open,
        'high': p.high,
        'low': p.low,
        'close': p.close,
        'volume': p.volume
    } for p in prices])
    prices_df.set_index('date', inplace=True)
    
    # Run backtest
    engine = BacktestEngine(prices_df, initial_capital)
    results = engine.run_backtest(strategy_type, parameters)
    
    # Save backtest results
    backtest = Backtest(
        strategy_id=strategy_id,
        symbol=symbol.upper(),
        start_date=prices_df.index[0],
        end_date=prices_df.index[-1],
        initial_capital=initial_capital,
        total_return=results['total_return'],
        annualized_return=results['annualized_return'],
        sharpe_ratio=results['sharpe_ratio'],
        max_drawdown=results['max_drawdown'],
        win_rate=results['win_rate'],
        total_trades=results['total_trades'],
        profitable_trades=results['profitable_trades'],
        avg_trade_return=results['avg_trade_return'],
        max_consecutive_wins=results['max_consecutive_wins'],
        max_consecutive_losses=results['max_consecutive_losses'],
        profit_factor=results['profit_factor'],
        equity_curve=results['equity_curve'],
        drawdown_curve=results['drawdown_curve'],
        monthly_returns=results['monthly_returns'],
        trades=results['trades'],
        execution_time_ms=results['execution_time_ms']
    )
    
    if strategy_id:
        db.session.add(backtest)
        db.session.commit()
        results['backtest_id'] = backtest.id
    
    return jsonify(results)


# Backtest History
@app.route('/api/backtests', methods=['GET'])
def get_backtests():
    """Get backtest history with optional filtering"""
    strategy_id = request.args.get('strategy_id')
    symbol = request.args.get('symbol')
    limit = request.args.get('limit', 50, type=int)
    
    query = Backtest.query
    
    if strategy_id:
        query = query.filter_by(strategy_id=strategy_id)
    if symbol:
        query = query.filter_by(symbol=symbol.upper())
    
    backtests = query.order_by(Backtest.executed_at.desc()).limit(limit).all()
    return jsonify([b.to_dict() for b in backtests])


@app.route('/api/backtests/<int:id>', methods=['GET'])
def get_backtest(id):
    """Get a specific backtest result"""
    backtest = Backtest.query.get_or_404(id)
    return jsonify(backtest.to_dict())


# =============================================================================
# SQL REPORTING ENDPOINTS
# =============================================================================

@app.route('/api/reports/strategy-comparison', methods=['GET'])
def strategy_comparison():
    """
    Complex SQL query to compare strategy performance across multiple dimensions.
    Demonstrates: JOIN, GROUP BY, aggregate functions, window functions simulation
    """
    sql = """
    SELECT 
        s.id as strategy_id,
        s.name as strategy_name,
        s.strategy_type,
        COUNT(b.id) as total_backtests,
        AVG(b.total_return) as avg_return,
        AVG(b.sharpe_ratio) as avg_sharpe,
        AVG(b.max_drawdown) as avg_max_drawdown,
        AVG(b.win_rate) as avg_win_rate,
        SUM(b.total_trades) as total_trades,
        MAX(b.total_return) as best_return,
        MIN(b.total_return) as worst_return,
        AVG(b.profit_factor) as avg_profit_factor
    FROM strategies s
    LEFT JOIN backtests b ON s.id = b.strategy_id
    GROUP BY s.id, s.name, s.strategy_type
    HAVING COUNT(b.id) > 0
    ORDER BY avg_return DESC
    """
    
    result = db.session.execute(db.text(sql))
    
    data = [{
        'strategy_id': row[0],
        'strategy_name': row[1],
        'strategy_type': row[2],
        'total_backtests': row[3],
        'avg_return': round(row[4], 2) if row[4] else 0,
        'avg_sharpe': round(row[5], 2) if row[5] else 0,
        'avg_max_drawdown': round(row[6], 2) if row[6] else 0,
        'avg_win_rate': round(row[7], 2) if row[7] else 0,
        'total_trades': row[8] or 0,
        'best_return': round(row[9], 2) if row[9] else 0,
        'worst_return': round(row[10], 2) if row[10] else 0,
        'avg_profit_factor': round(row[11], 2) if row[11] else 0
    } for row in result]
    
    return jsonify(data)


@app.route('/api/reports/performance-by-symbol', methods=['GET'])
def performance_by_symbol():
    """
    Analyze strategy performance across different symbols.
    Demonstrates: GROUP BY multiple columns, CASE statements simulation
    """
    sql = """
    SELECT 
        b.symbol,
        s.strategy_type,
        COUNT(b.id) as backtest_count,
        AVG(b.total_return) as avg_return,
        AVG(b.sharpe_ratio) as avg_sharpe,
        SUM(CASE WHEN b.total_return > 0 THEN 1 ELSE 0 END) as profitable_runs,
        AVG(b.total_trades) as avg_trades
    FROM backtests b
    JOIN strategies s ON b.strategy_id = s.id
    GROUP BY b.symbol, s.strategy_type
    ORDER BY b.symbol, avg_return DESC
    """
    
    result = db.session.execute(db.text(sql))
    
    data = [{
        'symbol': row[0],
        'strategy_type': row[1],
        'backtest_count': row[2],
        'avg_return': round(row[3], 2) if row[3] else 0,
        'avg_sharpe': round(row[4], 2) if row[4] else 0,
        'profitable_runs': row[5],
        'avg_trades': round(row[6], 1) if row[6] else 0
    } for row in result]
    
    return jsonify(data)


@app.route('/api/reports/time-analysis', methods=['GET'])
def time_analysis():
    """
    Analyze backtest performance over time periods.
    Demonstrates: Date functions, temporal grouping, trend analysis
    """
    sql = """
    SELECT 
        strftime('%Y-%m', b.executed_at) as month,
        COUNT(b.id) as backtests_run,
        AVG(b.total_return) as avg_return,
        AVG(b.sharpe_ratio) as avg_sharpe,
        SUM(b.total_trades) as total_trades,
        AVG(b.win_rate) as avg_win_rate
    FROM backtests b
    GROUP BY strftime('%Y-%m', b.executed_at)
    ORDER BY month DESC
    LIMIT 12
    """
    
    result = db.session.execute(db.text(sql))
    
    data = [{
        'month': row[0],
        'backtests_run': row[1],
        'avg_return': round(row[2], 2) if row[2] else 0,
        'avg_sharpe': round(row[3], 2) if row[3] else 0,
        'total_trades': row[4] or 0,
        'avg_win_rate': round(row[5], 2) if row[5] else 0
    } for row in result]
    
    return jsonify(data)


@app.route('/api/reports/top-performers', methods=['GET'])
def top_performers():
    """
    Get top performing backtests with full details.
    Demonstrates: Complex ORDER BY, LIMIT, JOIN for enriched data
    """
    metric = request.args.get('metric', 'total_return')
    limit = request.args.get('limit', 10, type=int)
    
    valid_metrics = ['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor']
    if metric not in valid_metrics:
        metric = 'total_return'
    
    sql = f"""
    SELECT 
        b.id,
        s.name as strategy_name,
        s.strategy_type,
        b.symbol,
        b.start_date,
        b.end_date,
        b.total_return,
        b.annualized_return,
        b.sharpe_ratio,
        b.max_drawdown,
        b.win_rate,
        b.total_trades,
        b.profit_factor,
        b.executed_at
    FROM backtests b
    JOIN strategies s ON b.strategy_id = s.id
    ORDER BY b.{metric} DESC
    LIMIT {limit}
    """
    
    result = db.session.execute(db.text(sql))
    
    data = [{
        'id': row[0],
        'strategy_name': row[1],
        'strategy_type': row[2],
        'symbol': row[3],
        'start_date': row[4],
        'end_date': row[5],
        'total_return': round(row[6], 2) if row[6] else 0,
        'annualized_return': round(row[7], 2) if row[7] else 0,
        'sharpe_ratio': round(row[8], 2) if row[8] else 0,
        'max_drawdown': round(row[9], 2) if row[9] else 0,
        'win_rate': round(row[10], 2) if row[10] else 0,
        'total_trades': row[11] or 0,
        'profit_factor': round(row[12], 2) if row[12] else 0,
        'executed_at': row[13]
    } for row in result]
    
    return jsonify(data)


@app.route('/api/reports/risk-metrics', methods=['GET'])
def risk_metrics():
    """
    Comprehensive risk analysis across strategies.
    Demonstrates: Statistical aggregations, risk-adjusted metrics
    """
    sql = """
    SELECT 
        s.strategy_type,
        COUNT(b.id) as sample_size,
        AVG(b.max_drawdown) as avg_drawdown,
        MIN(b.max_drawdown) as worst_drawdown,
        AVG(b.sharpe_ratio) as avg_sharpe,
        AVG(b.total_return / NULLIF(ABS(b.max_drawdown), 0)) as return_to_drawdown,
        AVG(b.win_rate) as avg_win_rate,
        AVG(b.max_consecutive_losses) as avg_losing_streak
    FROM backtests b
    JOIN strategies s ON b.strategy_id = s.id
    GROUP BY s.strategy_type
    HAVING COUNT(b.id) >= 1
    ORDER BY avg_sharpe DESC
    """
    
    result = db.session.execute(db.text(sql))
    
    data = [{
        'strategy_type': row[0],
        'sample_size': row[1],
        'avg_drawdown': round(row[2], 2) if row[2] else 0,
        'worst_drawdown': round(row[3], 2) if row[3] else 0,
        'avg_sharpe': round(row[4], 2) if row[4] else 0,
        'return_to_drawdown': round(row[5], 2) if row[5] else 0,
        'avg_win_rate': round(row[6], 2) if row[6] else 0,
        'avg_losing_streak': round(row[7], 1) if row[7] else 0
    } for row in result]
    
    return jsonify(data)


@app.route('/api/reports/dashboard-summary', methods=['GET'])
def dashboard_summary():
    """
    Aggregate summary for main dashboard.
    Demonstrates: Multiple aggregations in single query
    """
    sql = """
    SELECT 
        (SELECT COUNT(*) FROM strategies WHERE is_active = 1) as active_strategies,
        (SELECT COUNT(*) FROM backtests) as total_backtests,
        (SELECT COUNT(DISTINCT symbol) FROM backtests) as symbols_tested,
        (SELECT AVG(total_return) FROM backtests) as avg_return,
        (SELECT AVG(sharpe_ratio) FROM backtests) as avg_sharpe,
        (SELECT MAX(total_return) FROM backtests) as best_return,
        (SELECT SUM(total_trades) FROM backtests) as total_trades,
        (SELECT AVG(win_rate) FROM backtests) as avg_win_rate
    """
    
    result = db.session.execute(db.text(sql))
    row = result.fetchone()
    
    return jsonify({
        'active_strategies': row[0] or 0,
        'total_backtests': row[1] or 0,
        'symbols_tested': row[2] or 0,
        'avg_return': round(row[3], 2) if row[3] else 0,
        'avg_sharpe': round(row[4], 2) if row[4] else 0,
        'best_return': round(row[5], 2) if row[5] else 0,
        'total_trades': row[6] or 0,
        'avg_win_rate': round(row[7], 2) if row[7] else 0
    })


# =============================================================================
# INITIALIZATION
# =============================================================================

def init_db():
    """Initialize database and seed with sample data"""
    with app.app_context():
        db.create_all()
        seed_database()
        
        # Create sample strategies if none exist
        if Strategy.query.count() == 0:
            sample_strategies = [
                {
                    'name': 'Golden Cross',
                    'description': 'Classic 50/200 SMA crossover strategy',
                    'strategy_type': 'ma_crossover',
                    'parameters': {'fast_period': 50, 'slow_period': 200, 'ma_type': 'sma'}
                },
                {
                    'name': 'RSI Oversold Bounce',
                    'description': 'Buy oversold conditions, sell overbought',
                    'strategy_type': 'rsi',
                    'parameters': {'period': 14, 'oversold': 30, 'overbought': 70}
                },
                {
                    'name': 'MACD Momentum',
                    'description': 'Standard MACD crossover signals',
                    'strategy_type': 'macd',
                    'parameters': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
                },
                {
                    'name': 'Bollinger Bounce',
                    'description': 'Mean reversion using Bollinger Bands',
                    'strategy_type': 'bollinger',
                    'parameters': {'period': 20, 'std_dev': 2.0}
                }
            ]
            
            for s in sample_strategies:
                strategy = Strategy(**s)
                db.session.add(strategy)
            
            db.session.commit()
            print("Sample strategies created")


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
