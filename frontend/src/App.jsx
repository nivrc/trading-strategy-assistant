import React, { useState, useEffect } from 'react';
import './styles/App.css';
import Dashboard from './components/Dashboard';
import StrategyBuilder from './components/StrategyBuilder';
import BacktestResults from './components/BacktestResults';
import StrategyLibrary from './components/StrategyLibrary';
import Reports from './components/Reports';

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [templates, setTemplates] = useState({});
  const [symbols, setSymbols] = useState([]);
  const [strategies, setStrategies] = useState([]);
  const [backtestResult, setBacktestResult] = useState(null);
  const [dashboardSummary, setDashboardSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [notification, setNotification] = useState(null);

  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    try {
      const [templatesRes, symbolsRes, strategiesRes, summaryRes] = await Promise.all([
        fetch(`${API_BASE}/templates`),
        fetch(`${API_BASE}/symbols`),
        fetch(`${API_BASE}/strategies`),
        fetch(`${API_BASE}/reports/dashboard-summary`)
      ]);

      setTemplates(await templatesRes.json());
      setSymbols(await symbolsRes.json());
      setStrategies(await strategiesRes.json());
      setDashboardSummary(await summaryRes.json());
    } catch (error) {
      showNotification('Failed to load data. Is the backend running?', 'error');
    }
  };

  const showNotification = (message, type = 'success') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 4000);
  };

  const runBacktest = async (config) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (!response.ok) throw new Error('Backtest failed');

      const result = await response.json();
      setBacktestResult(result);
      setActiveTab('results');
      showNotification(`Backtest completed in ${result.execution_time_ms}ms`);
      
      // Refresh dashboard data
      const summaryRes = await fetch(`${API_BASE}/reports/dashboard-summary`);
      setDashboardSummary(await summaryRes.json());
      
      // Refresh strategies list
      const strategiesRes = await fetch(`${API_BASE}/strategies`);
      setStrategies(await strategiesRes.json());
    } catch (error) {
      showNotification('Backtest failed: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const saveStrategy = async (strategy) => {
    try {
      const response = await fetch(`${API_BASE}/strategies`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(strategy)
      });

      if (!response.ok) throw new Error('Failed to save strategy');

      const newStrategy = await response.json();
      setStrategies([newStrategy, ...strategies]);
      showNotification('Strategy saved successfully');
      return newStrategy;
    } catch (error) {
      showNotification('Failed to save strategy: ' + error.message, 'error');
      return null;
    }
  };

  const deleteStrategy = async (id) => {
    try {
      const response = await fetch(`${API_BASE}/strategies/${id}`, {
        method: 'DELETE'
      });

      if (!response.ok) throw new Error('Failed to delete strategy');

      setStrategies(strategies.filter(s => s.id !== id));
      showNotification('Strategy deleted');
    } catch (error) {
      showNotification('Failed to delete strategy: ' + error.message, 'error');
    }
  };

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: '◉' },
    { id: 'builder', label: 'Strategy Builder', icon: '⚡' },
    { id: 'library', label: 'Strategy Library', icon: '◫' },
    { id: 'results', label: 'Backtest Results', icon: '◈' },
    { id: 'reports', label: 'SQL Reports', icon: '◇' }
  ];

  return (
    <div className="app">
      {/* Notification */}
      {notification && (
        <div className={`notification ${notification.type}`}>
          <span>{notification.message}</span>
          <button onClick={() => setNotification(null)}>×</button>
        </div>
      )}

      {/* Loading Overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner">
            <div className="spinner-ring"></div>
            <span>Running Backtest...</span>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="logo">
            <span className="logo-icon">◈</span>
            <span className="logo-text">STRATEX</span>
          </div>
          <span className="header-subtitle">Trading Strategy Assistant</span>
        </div>
        <nav className="header-nav">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="tab-icon">{tab.icon}</span>
              <span className="tab-label">{tab.label}</span>
            </button>
          ))}
        </nav>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {activeTab === 'dashboard' && (
          <Dashboard 
            summary={dashboardSummary} 
            strategies={strategies}
            onNavigate={setActiveTab}
          />
        )}
        
        {activeTab === 'builder' && (
          <StrategyBuilder
            templates={templates}
            symbols={symbols}
            onSave={saveStrategy}
            onBacktest={runBacktest}
          />
        )}
        
        {activeTab === 'library' && (
          <StrategyLibrary
            strategies={strategies}
            templates={templates}
            symbols={symbols}
            onBacktest={runBacktest}
            onDelete={deleteStrategy}
          />
        )}
        
        {activeTab === 'results' && (
          <BacktestResults result={backtestResult} />
        )}
        
        {activeTab === 'reports' && (
          <Reports />
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <span>Trading Strategy Assistant</span>
        <span className="footer-divider">•</span>
        <span>Built with Flask + React</span>
        <span className="footer-divider">•</span>
        <span>SQL-Powered Analytics</span>
      </footer>
    </div>
  );
}

export default App;
