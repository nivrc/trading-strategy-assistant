import React, { useState, useEffect } from 'react';

function StrategyBuilder({ templates, symbols, onSave, onBacktest }) {
  const [selectedTemplate, setSelectedTemplate] = useState('ma_crossover');
  const [strategyName, setStrategyName] = useState('');
  const [description, setDescription] = useState('');
  const [parameters, setParameters] = useState({});
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState(10000);
  const [saveAndRun, setSaveAndRun] = useState(true);

  // Set default dates (last 2 years)
  useEffect(() => {
    const end = new Date();
    const start = new Date();
    start.setFullYear(start.getFullYear() - 2);
    
    setEndDate(end.toISOString().split('T')[0]);
    setStartDate(start.toISOString().split('T')[0]);
  }, []);

  // Initialize parameters when template changes
  useEffect(() => {
    if (templates[selectedTemplate]) {
      const defaultParams = {};
      Object.entries(templates[selectedTemplate].parameters).forEach(([key, config]) => {
        defaultParams[key] = config.default;
      });
      setParameters(defaultParams);
      setStrategyName(`${templates[selectedTemplate].name} Strategy`);
      setDescription(templates[selectedTemplate].description);
    }
  }, [selectedTemplate, templates]);

  const handleParameterChange = (key, value, config) => {
    let parsedValue = value;
    
    if (config.type === 'int') {
      parsedValue = parseInt(value) || config.default;
      parsedValue = Math.max(config.min, Math.min(config.max, parsedValue));
    } else if (config.type === 'float') {
      parsedValue = parseFloat(value) || config.default;
      parsedValue = Math.max(config.min, Math.min(config.max, parsedValue));
    }
    
    setParameters({ ...parameters, [key]: parsedValue });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    let strategyId = null;
    
    if (saveAndRun) {
      const savedStrategy = await onSave({
        name: strategyName,
        description,
        strategy_type: selectedTemplate,
        parameters
      });
      
      if (savedStrategy) {
        strategyId = savedStrategy.id;
      }
    }
    
    await onBacktest({
      strategy_id: strategyId,
      strategy_type: selectedTemplate,
      parameters,
      symbol: selectedSymbol,
      start_date: startDate,
      end_date: endDate,
      initial_capital: initialCapital
    });
  };

  const template = templates[selectedTemplate];

  return (
    <div className="strategy-builder">
      <div className="page-header">
        <h1>Strategy Builder</h1>
        <p>Configure and backtest trading strategies using pre-built templates</p>
      </div>

      <form onSubmit={handleSubmit} className="builder-form">
        <div className="builder-grid">
          {/* Left Column - Strategy Configuration */}
          <div className="builder-section">
            <h2>
              <span className="section-icon">◈</span>
              Strategy Template
            </h2>
            
            <div className="template-selector">
              {Object.entries(templates).map(([key, tmpl]) => (
                <button
                  key={key}
                  type="button"
                  className={`template-option ${selectedTemplate === key ? 'selected' : ''}`}
                  onClick={() => setSelectedTemplate(key)}
                >
                  <span className="template-name">{tmpl.name}</span>
                  <span className="template-desc">{tmpl.description}</span>
                </button>
              ))}
            </div>

            <div className="form-group">
              <label>Strategy Name</label>
              <input
                type="text"
                value={strategyName}
                onChange={(e) => setStrategyName(e.target.value)}
                placeholder="Enter strategy name"
                required
              />
            </div>

            <div className="form-group">
              <label>Description</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe your strategy"
                rows={3}
              />
            </div>
          </div>

          {/* Right Column - Parameters */}
          <div className="builder-section">
            <h2>
              <span className="section-icon">⚡</span>
              Parameters
            </h2>
            
            {template && (
              <div className="parameters-grid">
                {Object.entries(template.parameters).map(([key, config]) => (
                  <div key={key} className="param-group">
                    <label>{config.description}</label>
                    {config.type === 'select' ? (
                      <select
                        value={parameters[key] || config.default}
                        onChange={(e) => handleParameterChange(key, e.target.value, config)}
                      >
                        {config.options.map(opt => (
                          <option key={opt} value={opt}>{opt.toUpperCase()}</option>
                        ))}
                      </select>
                    ) : (
                      <div className="param-input-group">
                        <input
                          type="number"
                          value={parameters[key] ?? config.default}
                          onChange={(e) => handleParameterChange(key, e.target.value, config)}
                          min={config.min}
                          max={config.max}
                          step={config.type === 'float' ? 0.1 : 1}
                        />
                        <span className="param-range">
                          {config.min} - {config.max}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            <h2 className="mt-lg">
              <span className="section-icon">◇</span>
              Backtest Settings
            </h2>

            <div className="settings-grid">
              <div className="form-group">
                <label>Symbol</label>
                <select
                  value={selectedSymbol}
                  onChange={(e) => setSelectedSymbol(e.target.value)}
                >
                  {symbols.map(sym => (
                    <option key={sym} value={sym}>{sym}</option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Initial Capital ($)</label>
                <input
                  type="number"
                  value={initialCapital}
                  onChange={(e) => setInitialCapital(parseInt(e.target.value) || 10000)}
                  min={1000}
                  step={1000}
                />
              </div>

              <div className="form-group">
                <label>Start Date</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  required
                />
              </div>

              <div className="form-group">
                <label>End Date</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  required
                />
              </div>
            </div>
          </div>
        </div>

        {/* Action Bar */}
        <div className="builder-actions">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={saveAndRun}
              onChange={(e) => setSaveAndRun(e.target.checked)}
            />
            Save strategy to library
          </label>
          <button type="submit" className="btn-primary btn-lg">
            <span className="btn-icon">▶</span>
            Run Backtest
          </button>
        </div>
      </form>

      {/* Strategy Info Panel */}
      {template && (
        <div className="strategy-info-panel">
          <h3>About {template.name}</h3>
          <p>{template.description}</p>
          <div className="info-details">
            <div className="info-item">
              <span className="info-label">Strategy Type:</span>
              <span className="info-value">{selectedTemplate.replace('_', ' ')}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Parameters:</span>
              <span className="info-value">{Object.keys(template.parameters).length} configurable</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default StrategyBuilder;
