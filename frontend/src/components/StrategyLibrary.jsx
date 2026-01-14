import React, { useState } from 'react';

function StrategyLibrary({ strategies, templates, symbols, onBacktest, onDelete }) {
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [backtestSymbol, setBacktestSymbol] = useState('SPY');
  const [startDate, setStartDate] = useState(() => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 2);
    return d.toISOString().split('T')[0];
  });
  const [endDate, setEndDate] = useState(() => new Date().toISOString().split('T')[0]);
  const [initialCapital, setInitialCapital] = useState(10000);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(null);

  const handleBacktest = (strategy) => {
    onBacktest({
      strategy_id: strategy.id,
      symbol: backtestSymbol,
      start_date: startDate,
      end_date: endDate,
      initial_capital: initialCapital
    });
  };

  const confirmDelete = (id) => {
    onDelete(id);
    setShowDeleteConfirm(null);
    if (selectedStrategy?.id === id) {
      setSelectedStrategy(null);
    }
  };

  const getStrategyIcon = (type) => {
    const icons = {
      'ma_crossover': '↗',
      'rsi': '◎',
      'macd': '≋',
      'bollinger': '⊞',
      'dual_momentum': '⇈',
      'mean_reversion': '↔'
    };
    return icons[type] || '◈';
  };

  const getStrategyColor = (type) => {
    const colors = {
      'ma_crossover': 'cyan',
      'rsi': 'purple',
      'macd': 'green',
      'bollinger': 'orange',
      'dual_momentum': 'blue',
      'mean_reversion': 'pink'
    };
    return colors[type] || 'cyan';
  };

  return (
    <div className="strategy-library">
      <div className="page-header">
        <h1>Strategy Library</h1>
        <p>Manage and backtest your saved trading strategies</p>
      </div>

      <div className="library-layout">
        {/* Strategy List */}
        <div className="strategy-list-section">
          <div className="section-header">
            <h2>Saved Strategies</h2>
            <span className="strategy-count">{strategies.length} strategies</span>
          </div>

          {strategies.length === 0 ? (
            <div className="empty-list">
              <span className="empty-icon">◫</span>
              <p>No strategies saved yet</p>
              <p className="empty-hint">Create a strategy in the Strategy Builder</p>
            </div>
          ) : (
            <div className="strategy-list">
              {strategies.map(strategy => (
                <div 
                  key={strategy.id}
                  className={`strategy-card ${selectedStrategy?.id === strategy.id ? 'selected' : ''}`}
                  onClick={() => setSelectedStrategy(strategy)}
                >
                  <div className={`strategy-icon-wrapper ${getStrategyColor(strategy.strategy_type)}`}>
                    <span className="strategy-icon">{getStrategyIcon(strategy.strategy_type)}</span>
                  </div>
                  <div className="strategy-details">
                    <h3>{strategy.name}</h3>
                    <span className="strategy-type-badge">
                      {strategy.strategy_type.replace('_', ' ')}
                    </span>
                  </div>
                  <div className="strategy-meta">
                    <span>{new Date(strategy.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Strategy Detail & Backtest Panel */}
        <div className="strategy-detail-section">
          {selectedStrategy ? (
            <>
              <div className="detail-header">
                <div className={`detail-icon ${getStrategyColor(selectedStrategy.strategy_type)}`}>
                  {getStrategyIcon(selectedStrategy.strategy_type)}
                </div>
                <div className="detail-title">
                  <h2>{selectedStrategy.name}</h2>
                  <span className="detail-type">
                    {templates[selectedStrategy.strategy_type]?.name || selectedStrategy.strategy_type}
                  </span>
                </div>
                <button 
                  className="btn-delete"
                  onClick={() => setShowDeleteConfirm(selectedStrategy.id)}
                >
                  ×
                </button>
              </div>

              {selectedStrategy.description && (
                <p className="detail-description">{selectedStrategy.description}</p>
              )}

              {/* Parameters Display */}
              <div className="detail-section">
                <h3>Parameters</h3>
                <div className="params-display">
                  {Object.entries(selectedStrategy.parameters).map(([key, value]) => (
                    <div key={key} className="param-item">
                      <span className="param-key">{key.replace('_', ' ')}</span>
                      <span className="param-value">{value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Backtest Configuration */}
              <div className="detail-section">
                <h3>Run Backtest</h3>
                <div className="backtest-config">
                  <div className="config-row">
                    <div className="config-field">
                      <label>Symbol</label>
                      <select 
                        value={backtestSymbol}
                        onChange={(e) => setBacktestSymbol(e.target.value)}
                      >
                        {symbols.map(s => (
                          <option key={s} value={s}>{s}</option>
                        ))}
                      </select>
                    </div>
                    <div className="config-field">
                      <label>Capital</label>
                      <input
                        type="number"
                        value={initialCapital}
                        onChange={(e) => setInitialCapital(parseInt(e.target.value) || 10000)}
                        min={1000}
                        step={1000}
                      />
                    </div>
                  </div>
                  <div className="config-row">
                    <div className="config-field">
                      <label>Start Date</label>
                      <input
                        type="date"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                      />
                    </div>
                    <div className="config-field">
                      <label>End Date</label>
                      <input
                        type="date"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                      />
                    </div>
                  </div>
                </div>
                <button 
                  className="btn-primary btn-block"
                  onClick={() => handleBacktest(selectedStrategy)}
                >
                  <span className="btn-icon">▶</span>
                  Run Backtest
                </button>
              </div>

              {/* Strategy Meta */}
              <div className="detail-meta">
                <span>Created: {new Date(selectedStrategy.created_at).toLocaleString()}</span>
                <span>Updated: {new Date(selectedStrategy.updated_at).toLocaleString()}</span>
              </div>
            </>
          ) : (
            <div className="no-selection">
              <span className="no-selection-icon">◈</span>
              <h3>Select a Strategy</h3>
              <p>Choose a strategy from the list to view details and run backtests</p>
            </div>
          )}
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Delete Strategy?</h3>
            <p>This will permanently delete the strategy and all associated backtest results.</p>
            <div className="modal-actions">
              <button 
                className="btn-secondary"
                onClick={() => setShowDeleteConfirm(null)}
              >
                Cancel
              </button>
              <button 
                className="btn-danger"
                onClick={() => confirmDelete(showDeleteConfirm)}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default StrategyLibrary;
