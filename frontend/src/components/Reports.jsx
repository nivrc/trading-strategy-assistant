import React, { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:5000/api';

function Reports() {
  const [activeReport, setActiveReport] = useState('comparison');
  const [reportData, setReportData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [topMetric, setTopMetric] = useState('total_return');
  const [topLimit, setTopLimit] = useState(10);

  const reports = [
    { id: 'comparison', name: 'Strategy Comparison', icon: '◫', endpoint: '/reports/strategy-comparison' },
    { id: 'symbol', name: 'Performance by Symbol', icon: '◇', endpoint: '/reports/performance-by-symbol' },
    { id: 'time', name: 'Time Analysis', icon: '◎', endpoint: '/reports/time-analysis' },
    { id: 'top', name: 'Top Performers', icon: '★', endpoint: '/reports/top-performers' },
    { id: 'risk', name: 'Risk Metrics', icon: '⚡', endpoint: '/reports/risk-metrics' }
  ];

  useEffect(() => {
    fetchReport();
  }, [activeReport, topMetric, topLimit]);

  const fetchReport = async () => {
    setLoading(true);
    try {
      const report = reports.find(r => r.id === activeReport);
      let url = `${API_BASE}${report.endpoint}`;
      
      if (activeReport === 'top') {
        url += `?metric=${topMetric}&limit=${topLimit}`;
      }
      
      const response = await fetch(url);
      const data = await response.json();
      setReportData(data);
    } catch (error) {
      console.error('Failed to fetch report:', error);
      setReportData([]);
    } finally {
      setLoading(false);
    }
  };

  const renderStrategyComparison = () => {
    if (!reportData || reportData.length === 0) {
      return <div className="no-data">No data available. Run some backtests first!</div>;
    }

    return (
      <div className="report-table-wrapper">
        <table className="report-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Type</th>
              <th>Backtests</th>
              <th>Avg Return</th>
              <th>Avg Sharpe</th>
              <th>Avg Drawdown</th>
              <th>Win Rate</th>
              <th>Best/Worst</th>
            </tr>
          </thead>
          <tbody>
            {reportData.map((row, i) => (
              <tr key={i}>
                <td className="strategy-name">{row.strategy_name}</td>
                <td>
                  <span className="type-badge">{row.strategy_type.replace('_', ' ')}</span>
                </td>
                <td>{row.total_backtests}</td>
                <td className={row.avg_return >= 0 ? 'positive' : 'negative'}>
                  {row.avg_return >= 0 ? '+' : ''}{row.avg_return}%
                </td>
                <td className={row.avg_sharpe >= 0.5 ? 'positive' : 'neutral'}>
                  {row.avg_sharpe}
                </td>
                <td className="negative">{row.avg_max_drawdown}%</td>
                <td>{row.avg_win_rate}%</td>
                <td>
                  <span className="positive">+{row.best_return}%</span>
                  {' / '}
                  <span className="negative">{row.worst_return}%</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const renderSymbolPerformance = () => {
    if (!reportData || reportData.length === 0) {
      return <div className="no-data">No data available. Run some backtests first!</div>;
    }

    return (
      <div className="report-table-wrapper">
        <table className="report-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Strategy Type</th>
              <th>Backtests</th>
              <th>Avg Return</th>
              <th>Avg Sharpe</th>
              <th>Profitable Runs</th>
              <th>Avg Trades</th>
            </tr>
          </thead>
          <tbody>
            {reportData.map((row, i) => (
              <tr key={i}>
                <td className="symbol">{row.symbol}</td>
                <td>
                  <span className="type-badge">{row.strategy_type.replace('_', ' ')}</span>
                </td>
                <td>{row.backtest_count}</td>
                <td className={row.avg_return >= 0 ? 'positive' : 'negative'}>
                  {row.avg_return >= 0 ? '+' : ''}{row.avg_return}%
                </td>
                <td>{row.avg_sharpe}</td>
                <td>{row.profitable_runs}</td>
                <td>{row.avg_trades}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const renderTimeAnalysis = () => {
    if (!reportData || reportData.length === 0) {
      return <div className="no-data">No data available. Run some backtests first!</div>;
    }

    return (
      <div className="time-analysis">
        <div className="report-table-wrapper">
          <table className="report-table">
            <thead>
              <tr>
                <th>Month</th>
                <th>Backtests Run</th>
                <th>Avg Return</th>
                <th>Avg Sharpe</th>
                <th>Total Trades</th>
                <th>Avg Win Rate</th>
              </tr>
            </thead>
            <tbody>
              {reportData.map((row, i) => (
                <tr key={i}>
                  <td>{row.month}</td>
                  <td>{row.backtests_run}</td>
                  <td className={row.avg_return >= 0 ? 'positive' : 'negative'}>
                    {row.avg_return >= 0 ? '+' : ''}{row.avg_return}%
                  </td>
                  <td>{row.avg_sharpe}</td>
                  <td>{row.total_trades}</td>
                  <td>{row.avg_win_rate}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderTopPerformers = () => {
    if (!reportData || reportData.length === 0) {
      return <div className="no-data">No data available. Run some backtests first!</div>;
    }

    return (
      <div className="top-performers">
        <div className="top-filters">
          <div className="filter-group">
            <label>Sort By:</label>
            <select value={topMetric} onChange={(e) => setTopMetric(e.target.value)}>
              <option value="total_return">Total Return</option>
              <option value="sharpe_ratio">Sharpe Ratio</option>
              <option value="win_rate">Win Rate</option>
              <option value="profit_factor">Profit Factor</option>
            </select>
          </div>
          <div className="filter-group">
            <label>Show:</label>
            <select value={topLimit} onChange={(e) => setTopLimit(parseInt(e.target.value))}>
              <option value={5}>Top 5</option>
              <option value={10}>Top 10</option>
              <option value={20}>Top 20</option>
            </select>
          </div>
        </div>
        
        <div className="report-table-wrapper">
          <table className="report-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Strategy</th>
                <th>Symbol</th>
                <th>Return</th>
                <th>Sharpe</th>
                <th>Drawdown</th>
                <th>Win Rate</th>
                <th>Trades</th>
              </tr>
            </thead>
            <tbody>
              {reportData.map((row, i) => (
                <tr key={i} className={i < 3 ? 'top-three' : ''}>
                  <td className="rank">
                    {i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : i + 1}
                  </td>
                  <td className="strategy-name">{row.strategy_name}</td>
                  <td className="symbol">{row.symbol}</td>
                  <td className={row.total_return >= 0 ? 'positive' : 'negative'}>
                    {row.total_return >= 0 ? '+' : ''}{row.total_return}%
                  </td>
                  <td>{row.sharpe_ratio}</td>
                  <td className="negative">{row.max_drawdown}%</td>
                  <td>{row.win_rate}%</td>
                  <td>{row.total_trades}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderRiskMetrics = () => {
    if (!reportData || reportData.length === 0) {
      return <div className="no-data">No data available. Run some backtests first!</div>;
    }

    return (
      <div className="risk-metrics">
        <div className="report-table-wrapper">
          <table className="report-table">
            <thead>
              <tr>
                <th>Strategy Type</th>
                <th>Sample Size</th>
                <th>Avg Drawdown</th>
                <th>Worst Drawdown</th>
                <th>Avg Sharpe</th>
                <th>Return/Drawdown</th>
                <th>Avg Losing Streak</th>
              </tr>
            </thead>
            <tbody>
              {reportData.map((row, i) => (
                <tr key={i}>
                  <td>
                    <span className="type-badge">{row.strategy_type.replace('_', ' ')}</span>
                  </td>
                  <td>{row.sample_size}</td>
                  <td className="negative">{row.avg_drawdown}%</td>
                  <td className="negative">{row.worst_drawdown}%</td>
                  <td className={row.avg_sharpe >= 0.5 ? 'positive' : 'neutral'}>
                    {row.avg_sharpe}
                  </td>
                  <td className={row.return_to_drawdown > 0 ? 'positive' : 'negative'}>
                    {row.return_to_drawdown}x
                  </td>
                  <td>{row.avg_losing_streak}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div className="risk-summary">
          <h3>Risk Analysis Summary</h3>
          <div className="risk-cards">
            {reportData.map((row, i) => (
              <div key={i} className="risk-card">
                <h4>{row.strategy_type.replace('_', ' ')}</h4>
                <div className="risk-indicator">
                  <div 
                    className="risk-bar"
                    style={{ 
                      width: `${Math.min(100, Math.abs(row.worst_drawdown))}%`,
                      backgroundColor: Math.abs(row.worst_drawdown) > 30 ? '#ff6b6b' : 
                                       Math.abs(row.worst_drawdown) > 20 ? '#ffd93d' : '#00d4aa'
                    }}
                  ></div>
                </div>
                <span className="risk-label">
                  {Math.abs(row.worst_drawdown) > 30 ? 'High Risk' : 
                   Math.abs(row.worst_drawdown) > 20 ? 'Medium Risk' : 'Low Risk'}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderReport = () => {
    switch (activeReport) {
      case 'comparison': return renderStrategyComparison();
      case 'symbol': return renderSymbolPerformance();
      case 'time': return renderTimeAnalysis();
      case 'top': return renderTopPerformers();
      case 'risk': return renderRiskMetrics();
      default: return null;
    }
  };

  // SQL Query Display
  const sqlQueries = {
    comparison: `SELECT 
    s.name as strategy_name,
    s.strategy_type,
    COUNT(b.id) as total_backtests,
    AVG(b.total_return) as avg_return,
    AVG(b.sharpe_ratio) as avg_sharpe,
    AVG(b.max_drawdown) as avg_max_drawdown,
    AVG(b.win_rate) as avg_win_rate,
    MAX(b.total_return) as best_return,
    MIN(b.total_return) as worst_return
FROM strategies s
LEFT JOIN backtests b ON s.id = b.strategy_id
GROUP BY s.id, s.name, s.strategy_type
HAVING COUNT(b.id) > 0
ORDER BY avg_return DESC`,
    symbol: `SELECT 
    b.symbol,
    s.strategy_type,
    COUNT(b.id) as backtest_count,
    AVG(b.total_return) as avg_return,
    AVG(b.sharpe_ratio) as avg_sharpe,
    SUM(CASE WHEN b.total_return > 0 THEN 1 ELSE 0 END) as profitable_runs
FROM backtests b
JOIN strategies s ON b.strategy_id = s.id
GROUP BY b.symbol, s.strategy_type
ORDER BY b.symbol, avg_return DESC`,
    time: `SELECT 
    strftime('%Y-%m', b.executed_at) as month,
    COUNT(b.id) as backtests_run,
    AVG(b.total_return) as avg_return,
    AVG(b.sharpe_ratio) as avg_sharpe,
    SUM(b.total_trades) as total_trades
FROM backtests b
GROUP BY strftime('%Y-%m', b.executed_at)
ORDER BY month DESC
LIMIT 12`,
    top: `SELECT 
    b.id, s.name, b.symbol,
    b.total_return, b.sharpe_ratio,
    b.max_drawdown, b.win_rate,
    b.total_trades
FROM backtests b
JOIN strategies s ON b.strategy_id = s.id
ORDER BY b.${topMetric} DESC
LIMIT ${topLimit}`,
    risk: `SELECT 
    s.strategy_type,
    COUNT(b.id) as sample_size,
    AVG(b.max_drawdown) as avg_drawdown,
    MIN(b.max_drawdown) as worst_drawdown,
    AVG(b.sharpe_ratio) as avg_sharpe,
    AVG(b.total_return / NULLIF(ABS(b.max_drawdown), 0)) as return_to_drawdown
FROM backtests b
JOIN strategies s ON b.strategy_id = s.id
GROUP BY s.strategy_type
ORDER BY avg_sharpe DESC`
  };

  return (
    <div className="reports">
      <div className="page-header">
        <h1>SQL Reports</h1>
        <p>Advanced analytics powered by complex SQL queries</p>
      </div>

      {/* Report Selector */}
      <div className="report-selector">
        {reports.map(report => (
          <button
            key={report.id}
            className={`report-tab ${activeReport === report.id ? 'active' : ''}`}
            onClick={() => setActiveReport(report.id)}
          >
            <span className="report-icon">{report.icon}</span>
            <span className="report-name">{report.name}</span>
          </button>
        ))}
      </div>

      {/* SQL Query Display */}
      <div className="sql-display">
        <div className="sql-header">
          <span className="sql-label">SQL Query</span>
          <span className="sql-badge">PostgreSQL Compatible</span>
        </div>
        <pre className="sql-code">{sqlQueries[activeReport]}</pre>
      </div>

      {/* Report Content */}
      <div className="report-content">
        {loading ? (
          <div className="loading-report">
            <div className="loading-spinner-small"></div>
            <span>Loading report...</span>
          </div>
        ) : (
          renderReport()
        )}
      </div>
    </div>
  );
}

export default Reports;
