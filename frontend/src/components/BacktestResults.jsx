import React, { useState } from 'react';

function BacktestResults({ result }) {
  const [activeChart, setActiveChart] = useState('equity');

  if (!result) {
    return (
      <div className="backtest-results">
        <div className="page-header">
          <h1>Backtest Results</h1>
          <p>Run a backtest to see performance analysis</p>
        </div>
        <div className="empty-state">
          <div className="empty-icon">◈</div>
          <h3>No Results Yet</h3>
          <p>Configure and run a backtest from the Strategy Builder or Library to see results here.</p>
        </div>
      </div>
    );
  }

  const metrics = [
    { 
      label: 'Total Return', 
      value: `${result.total_return >= 0 ? '+' : ''}${result.total_return}%`,
      positive: result.total_return >= 0
    },
    { 
      label: 'Annualized Return', 
      value: `${result.annualized_return >= 0 ? '+' : ''}${result.annualized_return}%`,
      positive: result.annualized_return >= 0
    },
    { 
      label: 'Sharpe Ratio', 
      value: result.sharpe_ratio.toFixed(2),
      positive: result.sharpe_ratio > 0.5
    },
    { 
      label: 'Max Drawdown', 
      value: `${result.max_drawdown}%`,
      positive: result.max_drawdown > -20
    },
    { 
      label: 'Win Rate', 
      value: `${result.win_rate}%`,
      positive: result.win_rate > 50
    },
    { 
      label: 'Total Trades', 
      value: result.total_trades,
      neutral: true
    },
    { 
      label: 'Profit Factor', 
      value: result.profit_factor === 999.99 ? '∞' : result.profit_factor.toFixed(2),
      positive: result.profit_factor > 1.5
    },
    { 
      label: 'Avg Trade Return', 
      value: `${result.avg_trade_return >= 0 ? '+' : ''}${result.avg_trade_return}%`,
      positive: result.avg_trade_return > 0
    }
  ];

  // Calculate chart dimensions
  const chartWidth = 800;
  const chartHeight = 300;
  const padding = { top: 20, right: 20, bottom: 40, left: 60 };
  const plotWidth = chartWidth - padding.left - padding.right;
  const plotHeight = chartHeight - padding.top - padding.bottom;

  // Equity curve rendering
  const renderEquityCurve = () => {
    if (!result.equity_curve || result.equity_curve.length === 0) return null;

    const data = result.equity_curve;
    const minEquity = Math.min(...data.map(d => d.equity));
    const maxEquity = Math.max(...data.map(d => d.equity));
    const range = maxEquity - minEquity || 1;

    const points = data.map((d, i) => {
      const x = padding.left + (i / (data.length - 1)) * plotWidth;
      const y = padding.top + plotHeight - ((d.equity - minEquity) / range) * plotHeight;
      return `${x},${y}`;
    }).join(' ');

    const areaPoints = `${padding.left},${padding.top + plotHeight} ${points} ${padding.left + plotWidth},${padding.top + plotHeight}`;

    return (
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="chart-svg">
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((p, i) => (
          <g key={i}>
            <line
              x1={padding.left}
              y1={padding.top + plotHeight * (1 - p)}
              x2={padding.left + plotWidth}
              y2={padding.top + plotHeight * (1 - p)}
              stroke="rgba(255,255,255,0.1)"
              strokeDasharray="4,4"
            />
            <text
              x={padding.left - 10}
              y={padding.top + plotHeight * (1 - p)}
              textAnchor="end"
              fill="rgba(255,255,255,0.5)"
              fontSize="11"
              dominantBaseline="middle"
            >
              ${Math.round(minEquity + range * p).toLocaleString()}
            </text>
          </g>
        ))}
        
        {/* Area fill */}
        <polygon
          points={areaPoints}
          fill="url(#equityGradient)"
        />
        
        {/* Line */}
        <polyline
          points={points}
          fill="none"
          stroke="#00d4aa"
          strokeWidth="2"
        />
        
        {/* Gradient definition */}
        <defs>
          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(0, 212, 170, 0.3)" />
            <stop offset="100%" stopColor="rgba(0, 212, 170, 0)" />
          </linearGradient>
        </defs>
      </svg>
    );
  };

  // Drawdown curve rendering
  const renderDrawdownCurve = () => {
    if (!result.drawdown_curve || result.drawdown_curve.length === 0) return null;

    const data = result.drawdown_curve;
    const minDrawdown = Math.min(...data.map(d => d.drawdown));

    const points = data.map((d, i) => {
      const x = padding.left + (i / (data.length - 1)) * plotWidth;
      const y = padding.top + (Math.abs(d.drawdown) / Math.abs(minDrawdown || 1)) * plotHeight;
      return `${x},${y}`;
    }).join(' ');

    const areaPoints = `${padding.left},${padding.top} ${points} ${padding.left + plotWidth},${padding.top}`;

    return (
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="chart-svg">
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((p, i) => (
          <g key={i}>
            <line
              x1={padding.left}
              y1={padding.top + plotHeight * p}
              x2={padding.left + plotWidth}
              y2={padding.top + plotHeight * p}
              stroke="rgba(255,255,255,0.1)"
              strokeDasharray="4,4"
            />
            <text
              x={padding.left - 10}
              y={padding.top + plotHeight * p}
              textAnchor="end"
              fill="rgba(255,255,255,0.5)"
              fontSize="11"
              dominantBaseline="middle"
            >
              {Math.round(minDrawdown * p)}%
            </text>
          </g>
        ))}
        
        {/* Area fill */}
        <polygon
          points={areaPoints}
          fill="url(#drawdownGradient)"
        />
        
        {/* Line */}
        <polyline
          points={points}
          fill="none"
          stroke="#ff6b6b"
          strokeWidth="2"
        />
        
        <defs>
          <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(255, 107, 107, 0)" />
            <stop offset="100%" stopColor="rgba(255, 107, 107, 0.3)" />
          </linearGradient>
        </defs>
      </svg>
    );
  };

  // Monthly returns rendering
  const renderMonthlyReturns = () => {
    if (!result.monthly_returns || result.monthly_returns.length === 0) return null;

    const data = result.monthly_returns;
    const maxReturn = Math.max(...data.map(d => Math.abs(d.return)));
    const barWidth = Math.max(10, (plotWidth / data.length) - 4);

    return (
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="chart-svg">
        {/* Zero line */}
        <line
          x1={padding.left}
          y1={padding.top + plotHeight / 2}
          x2={padding.left + plotWidth}
          y2={padding.top + plotHeight / 2}
          stroke="rgba(255,255,255,0.3)"
          strokeWidth="1"
        />
        
        {/* Bars */}
        {data.map((d, i) => {
          const x = padding.left + (i / data.length) * plotWidth + 2;
          const barHeight = (Math.abs(d.return) / (maxReturn || 1)) * (plotHeight / 2);
          const y = d.return >= 0 
            ? padding.top + plotHeight / 2 - barHeight
            : padding.top + plotHeight / 2;
          
          return (
            <g key={i}>
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                fill={d.return >= 0 ? '#00d4aa' : '#ff6b6b'}
                rx="2"
              />
            </g>
          );
        })}
        
        {/* Y-axis labels */}
        <text
          x={padding.left - 10}
          y={padding.top}
          textAnchor="end"
          fill="rgba(255,255,255,0.5)"
          fontSize="11"
        >
          +{Math.round(maxReturn)}%
        </text>
        <text
          x={padding.left - 10}
          y={padding.top + plotHeight}
          textAnchor="end"
          fill="rgba(255,255,255,0.5)"
          fontSize="11"
        >
          -{Math.round(maxReturn)}%
        </text>
      </svg>
    );
  };

  return (
    <div className="backtest-results">
      <div className="page-header">
        <h1>Backtest Results</h1>
        <p>Executed in {result.execution_time_ms}ms</p>
      </div>

      {/* Metrics Grid */}
      <div className="results-metrics">
        {metrics.map((metric, i) => (
          <div key={i} className="result-metric">
            <span className={`metric-value ${metric.neutral ? '' : metric.positive ? 'positive' : 'negative'}`}>
              {metric.value}
            </span>
            <span className="metric-label">{metric.label}</span>
          </div>
        ))}
      </div>

      {/* Charts Section */}
      <div className="charts-section">
        <div className="chart-tabs">
          <button 
            className={`chart-tab ${activeChart === 'equity' ? 'active' : ''}`}
            onClick={() => setActiveChart('equity')}
          >
            Equity Curve
          </button>
          <button 
            className={`chart-tab ${activeChart === 'drawdown' ? 'active' : ''}`}
            onClick={() => setActiveChart('drawdown')}
          >
            Drawdown
          </button>
          <button 
            className={`chart-tab ${activeChart === 'monthly' ? 'active' : ''}`}
            onClick={() => setActiveChart('monthly')}
          >
            Monthly Returns
          </button>
        </div>

        <div className="chart-container">
          {activeChart === 'equity' && renderEquityCurve()}
          {activeChart === 'drawdown' && renderDrawdownCurve()}
          {activeChart === 'monthly' && renderMonthlyReturns()}
        </div>
      </div>

      {/* Trade List */}
      {result.trades && result.trades.length > 0 && (
        <div className="trades-section">
          <h2>Trade History</h2>
          <div className="trades-table-wrapper">
            <table className="trades-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Entry Date</th>
                  <th>Exit Date</th>
                  <th>Entry Price</th>
                  <th>Exit Price</th>
                  <th>Return</th>
                  <th>Profit</th>
                </tr>
              </thead>
              <tbody>
                {result.trades.slice(0, 50).map((trade, i) => (
                  <tr key={i} className={trade.return_pct >= 0 ? 'profitable' : 'losing'}>
                    <td>{i + 1}</td>
                    <td>{trade.entry_date}</td>
                    <td>{trade.exit_date}</td>
                    <td>${trade.entry_price.toFixed(2)}</td>
                    <td>${trade.exit_price.toFixed(2)}</td>
                    <td className={trade.return_pct >= 0 ? 'positive' : 'negative'}>
                      {trade.return_pct >= 0 ? '+' : ''}{trade.return_pct}%
                    </td>
                    <td className={trade.profit >= 0 ? 'positive' : 'negative'}>
                      {trade.profit >= 0 ? '+' : ''}${trade.profit.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {result.trades.length > 50 && (
              <p className="trades-note">Showing first 50 of {result.trades.length} trades</p>
            )}
          </div>
        </div>
      )}

      {/* Additional Stats */}
      <div className="additional-stats">
        <div className="stat-box">
          <h3>Winning Streaks</h3>
          <div className="streak-stats">
            <div className="streak-item">
              <span className="streak-value positive">{result.max_consecutive_wins}</span>
              <span className="streak-label">Max Wins</span>
            </div>
            <div className="streak-item">
              <span className="streak-value negative">{result.max_consecutive_losses}</span>
              <span className="streak-label">Max Losses</span>
            </div>
          </div>
        </div>
        <div className="stat-box">
          <h3>Trade Breakdown</h3>
          <div className="trade-breakdown">
            <div className="breakdown-bar">
              <div 
                className="breakdown-fill positive"
                style={{ width: `${result.win_rate}%` }}
              ></div>
            </div>
            <div className="breakdown-labels">
              <span className="positive">{result.profitable_trades} Winners</span>
              <span className="negative">{result.total_trades - result.profitable_trades} Losers</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default BacktestResults;
