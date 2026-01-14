import React from 'react';

function Dashboard({ summary, strategies, onNavigate }) {
  const stats = [
    { 
      label: 'Active Strategies', 
      value: summary?.active_strategies || 0, 
      icon: '◫',
      color: 'cyan'
    },
    { 
      label: 'Total Backtests', 
      value: summary?.total_backtests || 0, 
      icon: '◈',
      color: 'green'
    },
    { 
      label: 'Symbols Tested', 
      value: summary?.symbols_tested || 0, 
      icon: '◇',
      color: 'purple'
    },
    { 
      label: 'Total Trades', 
      value: summary?.total_trades?.toLocaleString() || 0, 
      icon: '⟡',
      color: 'orange'
    }
  ];

  const metrics = [
    { 
      label: 'Avg Return', 
      value: `${summary?.avg_return?.toFixed(1) || 0}%`,
      subtext: 'Across all backtests',
      positive: (summary?.avg_return || 0) > 0
    },
    { 
      label: 'Best Return', 
      value: `${summary?.best_return?.toFixed(1) || 0}%`,
      subtext: 'Single backtest max',
      positive: true
    },
    { 
      label: 'Avg Sharpe', 
      value: summary?.avg_sharpe?.toFixed(2) || '0.00',
      subtext: 'Risk-adjusted return',
      positive: (summary?.avg_sharpe || 0) > 0.5
    },
    { 
      label: 'Avg Win Rate', 
      value: `${summary?.avg_win_rate?.toFixed(1) || 0}%`,
      subtext: 'Profitable trades',
      positive: (summary?.avg_win_rate || 0) > 50
    }
  ];

  return (
    <div className="dashboard">
      <div className="page-header">
        <h1>Dashboard</h1>
        <p>Overview of your trading strategy performance</p>
      </div>

      {/* Stats Grid */}
      <div className="stats-grid">
        {stats.map((stat, i) => (
          <div key={i} className={`stat-card stat-${stat.color}`}>
            <div className="stat-icon">{stat.icon}</div>
            <div className="stat-content">
              <span className="stat-value">{stat.value}</span>
              <span className="stat-label">{stat.label}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Metrics Row */}
      <div className="metrics-section">
        <h2>Performance Metrics</h2>
        <div className="metrics-grid">
          {metrics.map((metric, i) => (
            <div key={i} className="metric-card">
              <span className={`metric-value ${metric.positive ? 'positive' : 'negative'}`}>
                {metric.value}
              </span>
              <span className="metric-label">{metric.label}</span>
              <span className="metric-subtext">{metric.subtext}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="quick-actions-section">
        <h2>Quick Actions</h2>
        <div className="action-cards">
          <button className="action-card" onClick={() => onNavigate('builder')}>
            <span className="action-icon">⚡</span>
            <span className="action-title">Create Strategy</span>
            <span className="action-desc">Build a new trading strategy from templates</span>
          </button>
          <button className="action-card" onClick={() => onNavigate('library')}>
            <span className="action-icon">◫</span>
            <span className="action-title">Run Backtest</span>
            <span className="action-desc">Test an existing strategy on historical data</span>
          </button>
          <button className="action-card" onClick={() => onNavigate('reports')}>
            <span className="action-icon">◇</span>
            <span className="action-title">View Reports</span>
            <span className="action-desc">Analyze performance with SQL-powered reports</span>
          </button>
        </div>
      </div>

      {/* Recent Strategies */}
      {strategies.length > 0 && (
        <div className="recent-section">
          <h2>Recent Strategies</h2>
          <div className="recent-strategies">
            {strategies.slice(0, 5).map(strategy => (
              <div key={strategy.id} className="strategy-row">
                <div className="strategy-info">
                  <span className="strategy-name">{strategy.name}</span>
                  <span className="strategy-type">{strategy.strategy_type.replace('_', ' ')}</span>
                </div>
                <div className="strategy-date">
                  {new Date(strategy.created_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {(!summary || summary.total_backtests === 0) && (
        <div className="empty-state">
          <div className="empty-icon">◈</div>
          <h3>No backtests yet</h3>
          <p>Create a strategy and run your first backtest to see performance data here.</p>
          <button className="btn-primary" onClick={() => onNavigate('builder')}>
            Create Your First Strategy
          </button>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
