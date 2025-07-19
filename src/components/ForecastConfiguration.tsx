import React from 'react';
import { TrendingUp, Calendar, Clock, Package, Users, MapPin, Target, Brain, Zap, Grid, List, X } from 'lucide-react';
import { ForecastConfig } from '../services/api';
import { ExternalFactorSelector } from './ExternalFactorSelector';

interface ForecastConfigurationProps {
  config: ForecastConfig;
  onChange: (config: ForecastConfig) => void;
  productOptions: string[];
  customerOptions: string[];
  locationOptions: string[];
}

export const ForecastConfiguration: React.FC<ForecastConfigurationProps> = ({
  config,
  onChange,
  productOptions,
  customerOptions,
  locationOptions
}) => {
  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [multiSelectMode, setMultiSelectMode] = React.useState(false);
  const [simpleMultiSelect, setSimpleMultiSelect] = React.useState(false);

  const algorithms = [
    // Statistical Methods
    { value: 'linear_regression', label: 'Linear Regression', icon: '📈', description: 'Simple trend-based forecasting', category: 'Statistical' },
    { value: 'polynomial_regression', label: 'Polynomial Regression', icon: '📊', description: 'Captures non-linear patterns', category: 'Statistical' },
    { value: 'exponential_smoothing', label: 'Exponential Smoothing', icon: '🌊', description: 'Weighted recent observations', category: 'Statistical' },
    { value: 'simple_exponential_smoothing', label: 'Simple Exponential Smoothing', icon: '🌀', description: 'Optimized exponential smoothing', category: 'Statistical' },
    { value: 'holt_winters', label: 'Holt-Winters', icon: '❄️', description: 'Handles trend and seasonality', category: 'Statistical' },
    { value: 'damped_trend', label: 'Damped Trend', icon: '📉', description: 'Exponential smoothing with damped trend', category: 'Statistical' },
    { value: 'arima', label: 'ARIMA', icon: '🔄', description: 'Autoregressive integrated model', category: 'Statistical' },
    { value: 'sarima', label: 'SARIMA', icon: '🔄', description: 'Seasonal ARIMA with better seasonality', category: 'Statistical' },
    { value: 'theta_method', label: 'Theta Method', icon: '🎯', description: 'Simple but effective statistical method', category: 'Statistical' },
    { value: 'drift_method', label: 'Drift Method', icon: '📈', description: 'Linear trend extrapolation', category: 'Statistical' },
    { value: 'naive_seasonal', label: 'Naive Seasonal', icon: '🔁', description: 'Simple seasonal pattern repetition', category: 'Statistical' },
    { value: 'prophet_like', label: 'Prophet-like', icon: '🔮', description: 'Trend + seasonality decomposition', category: 'Statistical' },
    
    // Machine Learning Methods
    { value: 'random_forest', label: 'Random Forest', icon: '🌳', description: 'Machine learning ensemble', category: 'Machine Learning' },
    { value: 'xgboost', label: 'XGBoost', icon: '🚀', description: 'Gradient boosting algorithm', category: 'Machine Learning' },
    { value: 'svr', label: 'Support Vector Regression', icon: '🎯', description: 'SVM for regression tasks', category: 'Machine Learning' },
    { value: 'knn', label: 'K-Nearest Neighbors', icon: '👥', description: 'Instance-based learning', category: 'Machine Learning' },
    { value: 'gaussian_process', label: 'Gaussian Process', icon: '📊', description: 'Probabilistic approach with uncertainty', category: 'Machine Learning' },
    { value: 'neural_network', label: 'Neural Network (MLP)', icon: '🧠', description: 'Multi-layer perceptron', category: 'Machine Learning' },
    { value: 'lstm_like', label: 'LSTM-like Network', icon: '🔗', description: 'Neural network with memory', category: 'Machine Learning' },
    
    // Specialized Methods
    { value: 'seasonal_decomposition', label: 'Seasonal Decomposition', icon: '🗓️', description: 'Separates trend and seasonality', category: 'Specialized' },
    { value: 'moving_average', label: 'Moving Average', icon: '📉', description: 'Smoothed historical average', category: 'Specialized' },
    { value: 'crostons_method', label: "Croston's Method", icon: '⚡', description: 'For intermittent/sparse demand', category: 'Specialized' },
    
    // Auto-Select (Featured)
    { value: 'best_fit', label: 'Best Fit (Auto-Select)', icon: '🎯', description: 'Automatically tests all 23 algorithms', featured: true, category: 'Auto-Select' }
  ];

  const handleChange = (field: keyof ForecastConfig, value: any) => {
    const newConfig = { ...config, [field]: value };
    
    // Reset selected item when forecast type changes
    if (field === 'forecastBy') {
      newConfig.selectedItem = '';
      newConfig.selectedItems = [];
      newConfig.selectedProduct = '';
      newConfig.selectedCustomer = '';
      newConfig.selectedLocation = '';
      newConfig.selectedProducts = [];
      newConfig.selectedCustomers = [];
      newConfig.selectedLocations = [];
    }
    
    onChange(newConfig);
  };

  const handleMultiSelectChange = (field: 'selectedProducts' | 'selectedCustomers' | 'selectedLocations', value: string) => {
    const currentValues = config[field] || [];
    const newValues = currentValues.includes(value)
      ? currentValues.filter(v => v !== value)
      : [...currentValues, value];
    
    handleChange(field, newValues);
  };

  const handleSimpleMultiSelectChange = (value: string) => {
    const currentValues = config.selectedItems || [];
    const newValues = currentValues.includes(value)
      ? currentValues.filter(v => v !== value)
      : [...currentValues, value];
    
    handleChange('selectedItems', newValues);
  };
  const toggleMultiSelectMode = () => {
    const newMode = !multiSelectMode;
    setMultiSelectMode(newMode);
    
    // Update config
    const newConfig = { 
      ...config, 
      multiSelect: newMode,
      // Clear conflicting selections when switching modes
      selectedItem: '',
      selectedItems: [],
      selectedProduct: '',
      selectedCustomer: '',
      selectedLocation: '',
      selectedProducts: [],
      selectedCustomers: [],
      selectedLocations: []
    };
    
    if (newMode) {
      // Switching to multi-select mode
      // Initialize empty arrays for multi-select
      newConfig.selectedProducts = [];
      newConfig.selectedCustomers = [];
      newConfig.selectedLocations = [];
      setShowAdvanced(true); // Force advanced mode for multi-select
      setSimpleMultiSelect(false); // Disable simple multi-select
      setShowAdvanced(true);
    }
    
    onChange(newConfig);
  };
  const toggleSimpleMultiSelect = () => {
    const newMode = !simpleMultiSelect;
    setSimpleMultiSelect(newMode);
    
    const newConfig = { ...config };
    
    if (newMode) {
      // Switching to simple multi-select
      newConfig.selectedItems = config.selectedItem ? [config.selectedItem] : [];
      newConfig.selectedItem = '';
      setMultiSelectMode(false); // Disable advanced multi-select
      setShowAdvanced(false); // Disable advanced mode
    } else {
      // Switching back to single select
      newConfig.selectedItems = [];
    }
    
    onChange(newConfig);
  };

  const getOptionsForType = () => {
    switch (config.forecastBy) {
      case 'product':
        return productOptions;
      case 'customer':
        return customerOptions;
      case 'location':
        return locationOptions;
      default:
        return [];
    }
  };

  const getForecastTypeIcon = () => {
    switch (config.forecastBy) {
      case 'product':
        return <Package className="w-4 h-4" />;
      case 'customer':
        return <Users className="w-4 h-4" />;
      case 'location':
        return <MapPin className="w-4 h-4" />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* Algorithm Selection */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center mb-6">
          <Brain className="w-6 h-6 text-purple-600 mr-2" />
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Select Forecasting Algorithm</h2>
            <p className="text-sm text-gray-600 mt-1">Choose from 23 advanced algorithms or use Best Fit for automatic selection</p>
          </div>
        </div>
        
        {/* Best Fit (Featured) */}
        <div className="mb-6">
          {algorithms.filter(alg => alg.featured).map((algorithm) => (
            <div
              key={algorithm.value}
              className={`relative p-6 border-2 rounded-xl cursor-pointer transition-all duration-200 ${
                config.algorithm === algorithm.value
                  ? 'border-purple-500 bg-gradient-to-br from-purple-50 to-indigo-50 shadow-lg'
                  : 'border-purple-300 bg-gradient-to-br from-purple-25 to-indigo-25 hover:border-purple-400'
              }`}
              onClick={() => handleChange('algorithm', algorithm.value)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="text-3xl animate-pulse">
                    {algorithm.icon}
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-purple-900 flex items-center">
                      {algorithm.label}
                      <Zap className="w-5 h-5 ml-2 text-purple-600" />
                    </h3>
                    <p className="text-purple-700 mt-1">
                      {algorithm.description}
                    </p>
                  </div>
                </div>
                {config.algorithm === algorithm.value && (
                  <div className="text-purple-500">
                    <div className="w-4 h-4 rounded-full bg-purple-500"></div>
                  </div>
                )}
              </div>
              <div className="absolute -top-2 -right-2 bg-purple-500 text-white text-xs px-3 py-1 rounded-full font-medium">
                RECOMMENDED
              </div>
            </div>
          ))}
        </div>
        
        {/* Algorithm Categories */}
        <div className="space-y-6">
          {['Statistical', 'Machine Learning', 'Specialized'].map((category) => (
            <div key={category}>
              <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                {category === 'Statistical' && <span className="mr-2">📊</span>}
                {category === 'Machine Learning' && <span className="mr-2">🤖</span>}
                {category === 'Specialized' && <span className="mr-2">⚙️</span>}
                {category} Methods
                <span className="ml-2 text-sm text-gray-500">
                  ({algorithms.filter(alg => alg.category === category).length} algorithms)
                </span>
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {algorithms.filter(alg => alg.category === category).map((algorithm) => (
                  <div
                    key={algorithm.value}
                    className={`relative p-3 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                      config.algorithm === algorithm.value
                        ? 'border-blue-500 bg-blue-50 shadow-md'
                        : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                    }`}
                    onClick={() => handleChange('algorithm', algorithm.value)}
                  >
                    <div className="flex items-start space-x-3">
                      <div className="text-xl">
                        {algorithm.icon}
                      </div>
                      <div className="flex-1">
                        <h4 className="font-medium text-gray-900 text-sm">
                          {algorithm.label}
                        </h4>
                        <p className="text-xs text-gray-600 mt-1">
                          {algorithm.description}
                        </p>
                      </div>
                    </div>
                    {config.algorithm === algorithm.value && (
                      <div className="absolute top-2 right-2">
                        <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
        
        {config.algorithm === 'best_fit' && (
          <div className="mt-4 p-4 bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <Target className="w-5 h-5 text-purple-600" />
              <div>
                <h4 className="font-medium text-purple-900">Best Fit Mode</h4>
                <p className="text-sm text-purple-700 mt-1">
                  This will run all 23 algorithms and automatically select the one with the highest accuracy for your data.
                  You'll see results from all algorithms for comparison.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Data Selection */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center mb-6">
          <TrendingUp className="w-6 h-6 text-blue-600 mr-2" />
          <h2 className="text-xl font-semibold text-gray-900">Data Selection</h2>
        </div>
        
        {/* Advanced Multi-Dimension Selection Toggle */}
        <div className="mb-6">
          <div className="space-y-4">
            {/* Multi-Selection Mode Toggle */}
            <div className="flex items-center justify-between p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border border-purple-200">
              <div className="flex items-center space-x-3">
                <Grid className="w-5 h-5 text-purple-600" />
                <div>
                  <h3 className="font-medium text-gray-900">Multi-Selection Forecasting</h3>
                  <p className="text-sm text-gray-600">Generate forecasts for multiple combinations</p>
                </div>
              </div>
              <button
                onClick={toggleMultiSelectMode}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  multiSelectMode 
                    ? 'bg-purple-600 text-white' 
                    : 'bg-white text-purple-600 border border-purple-600 hover:bg-purple-50'
                }`}
              >
                {multiSelectMode ? 'Single Mode' : 'Multi Mode'}
              </button>
            </div>
            
            {/* Simple Multi-Select Toggle (only show if not in advanced multi-select mode) */}
            {!multiSelectMode && (
              <div className="flex items-center justify-between p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200">
                <div className="flex items-center space-x-3">
                  <List className="w-5 h-5 text-green-600" />
                  <div>
                    <h3 className="font-medium text-gray-900">Simple Multi-Selection</h3>
                    <p className="text-sm text-gray-600">Select multiple items within the same dimension</p>
                  </div>
                </div>
                <button
                  onClick={toggleSimpleMultiSelect}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    simpleMultiSelect 
                      ? 'bg-green-600 text-white' 
                      : 'bg-white text-green-600 border border-green-600 hover:bg-green-50'
                  }`}
                >
                  {simpleMultiSelect ? 'Single Select' : 'Multi Select'}
                </button>
              </div>
            )}
            {/* Advanced Mode Toggle (only show if not in multi-select mode) */}
            {!multiSelectMode && !simpleMultiSelect && (
              <div className="flex items-center justify-between p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
                <div className="flex items-center space-x-3">
                  <Target className="w-5 h-5 text-blue-600" />
                  <div>
                    <h3 className="font-medium text-gray-900">Precise Forecasting</h3>
                    <p className="text-sm text-gray-600">Select specific Product + Customer + Location combination</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    showAdvanced 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-white text-blue-600 border border-blue-600 hover:bg-blue-50'
                  }`}
                >
                  {showAdvanced ? 'Simple Mode' : 'Advanced Mode'}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* External Factors Section */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <ExternalFactorSelector
            selectedFactors={config.externalFactors || []}
            onFactorsChange={(factors) => onChange({ ...config, externalFactors: factors })}
          />
        </div>

        {(showAdvanced || multiSelectMode) ? (
          /* Advanced/Multi-Selection Mode */
          <div className="space-y-4">
            {multiSelectMode && (
              <div className="flex items-center justify-between p-4 bg-gradient-to-r from-indigo-50 to-blue-50 rounded-lg border border-indigo-200">
                <div className="flex items-center space-x-3">
                  <Target className="w-5 h-5 text-indigo-600" />
                  <div>
                    <h3 className="font-medium text-gray-900">Advanced Mode (Precise Combinations)</h3>
                    <p className="text-sm text-gray-600">Generate forecasts for exact Product × Customer × Location combinations</p>
                  </div>
                </div>
                <button
                  onClick={toggleAdvancedMode}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    advancedMode 
                      ? 'bg-indigo-600 text-white' 
                      : 'bg-white text-indigo-600 border border-indigo-600 hover:bg-indigo-50'
                  }`}
                >
                  {advancedMode ? 'Flexible Mode' : 'Advanced Mode'}
                </button>
              </div>
            )}
            
            <div className="flex items-center space-x-2 mb-4">
              {advancedMode ? <Target className="w-5 h-5 text-indigo-600" /> : <Grid className="w-5 h-5 text-purple-600" />}
              <h3 className="text-lg font-semibold text-gray-900">
                {advancedMode ? 'Select Items for Precise Combinations' : 'Select Multiple Items (Flexible Combinations)'}
              </h3>
              {multiSelectMode && !advancedMode && (
                <p className="text-sm text-gray-600 ml-2">
                  Select at least 2 dimensions for combination forecasting
                </p>
              )}
              {advancedMode && (
                <p className="text-sm text-gray-600 ml-2">
                  All three dimensions required for precise forecasting
                </p>
              )}
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Product Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Package className="w-4 h-4 inline mr-1" />
                  Product{(multiSelectMode || advancedMode) ? 's' : ''}
                  {advancedMode && <span className="text-red-500 ml-1">*</span>}
                </label>
                
                {(multiSelectMode || advancedMode) ? (
                  <div className="border border-gray-300 rounded-lg p-3 max-h-40 overflow-y-auto">
                    <div className="space-y-2">
                      {filteredOptions.products.map((option) => (
                        <label key={option} className="flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={(config.selectedProducts || []).includes(option)}
                            onChange={() => handleMultiSelectChange('selectedProducts', option)}
                            className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                          />
                          <span className="text-sm text-gray-700">{option}</span>
                        </label>
                      ))}
                    </div>
                    {(config.selectedProducts || []).length > 0 && (
                      <div className="mt-2 pt-2 border-t border-gray-200">
                        <p className="text-xs text-purple-600 font-medium">
                          {(config.selectedProducts || []).length} selected
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <select
                    value={config.selectedProduct || ''}
                    onChange={(e) => handleChange('selectedProduct', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                  >
                    <option value="">Select Product</option>
                    {filteredOptions.products.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {/* Customer Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Users className="w-4 h-4 inline mr-1" />
                  Customer{(multiSelectMode || advancedMode) ? 's' : ''}
                  {advancedMode && <span className="text-red-500 ml-1">*</span>}
                </label>
                
                {(multiSelectMode || advancedMode) ? (
                  <div className="border border-gray-300 rounded-lg p-3 max-h-40 overflow-y-auto">
                    <div className="space-y-2">
                      {filteredOptions.customers.map((option) => (
                        <label key={option} className="flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={(config.selectedCustomers || []).includes(option)}
                            onChange={() => handleMultiSelectChange('selectedCustomers', option)}
                            className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                          />
                          <span className="text-sm text-gray-700">{option}</span>
                        </label>
                      ))}
                    </div>
                    {(config.selectedCustomers || []).length > 0 && (
                      <div className="mt-2 pt-2 border-t border-gray-200">
                        <p className="text-xs text-purple-600 font-medium">
                          {(config.selectedCustomers || []).length} selected
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <select
                    value={config.selectedCustomer || ''}
                    onChange={(e) => handleChange('selectedCustomer', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                  >
                    <option value="">Select Customer</option>
                    {filteredOptions.customers.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {/* Location Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <MapPin className="w-4 h-4 inline mr-1" />
                  Location{(multiSelectMode || advancedMode) ? 's' : ''}
                  {advancedMode && <span className="text-red-500 ml-1">*</span>}
                </label>
                
                {(multiSelectMode || advancedMode) ? (
                  <div className="border border-gray-300 rounded-lg p-3 max-h-40 overflow-y-auto">
                    <div className="space-y-2">
                      {filteredOptions.locations.map((option) => (
                        <label key={option} className="flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={(config.selectedLocations || []).includes(option)}
                            onChange={() => handleMultiSelectChange('selectedLocations', option)}
                            className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                          />
                          <span className="text-sm text-gray-700">{option}</span>
                        </label>
                      ))}
                    </div>
                    {(config.selectedLocations || []).length > 0 && (
                      <div className="mt-2 pt-2 border-t border-gray-200">
                        <p className="text-xs text-purple-600 font-medium">
                          {(config.selectedLocations || []).length} selected
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <select
                    value={config.selectedLocation || ''}
                    onChange={(e) => handleChange('selectedLocation', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                  >
                    <option value="">Select Location</option>
                    {filteredOptions.locations.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            </div>
            
            {multiSelectMode && !advancedMode && (
              <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                <div className="flex items-start space-x-3">
                  <Grid className="w-5 h-5 text-purple-600 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-purple-900">Multi-Selection Summary</h4>
                    <div className="text-sm text-purple-700 mt-1 space-y-1">
                      <p><strong>Products:</strong> {(config.selectedProducts || []).length} selected</p>
                      <p><strong>Customers:</strong> {(config.selectedCustomers || []).length} selected</p>
                      <p><strong>Locations:</strong> {(config.selectedLocations || []).length} selected</p>
                      <p><strong>Selected Dimensions:</strong> {
                        [(config.selectedProducts || []).length > 0 ? 'Products' : null,
                         (config.selectedCustomers || []).length > 0 ? 'Customers' : null,
                         (config.selectedLocations || []).length > 0 ? 'Locations' : null]
                        .filter(Boolean).join(', ') || 'None'
                      }</p>
                      <p><strong>Total Combinations:</strong> {
                        (() => {
                          const productCount = (config.selectedProducts || []).length || 1;
                          const customerCount = (config.selectedCustomers || []).length || 1;
                          const locationCount = (config.selectedLocations || []).length || 1;
                          
                          let total = 0;
                          if ((config.selectedProducts || []).length > 0 && (config.selectedCustomers || []).length > 0 && (config.selectedLocations || []).length > 0) {
                            total = productCount * customerCount * locationCount;
                          } else if ((config.selectedProducts || []).length > 0 && (config.selectedCustomers || []).length > 0) {
                            total = productCount * customerCount;
                          } else if ((config.selectedProducts || []).length > 0 && (config.selectedLocations || []).length > 0) {
                            total = productCount * locationCount;
                          } else if ((config.selectedCustomers || []).length > 0 && (config.selectedLocations || []).length > 0) {
                            total = customerCount * locationCount;
                          }
                          return total;
                        })()
                      }</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {advancedMode && (
              <div className="p-4 bg-indigo-50 border border-indigo-200 rounded-lg">
                <div className="flex items-start space-x-3">
                  <Target className="w-5 h-5 text-indigo-600 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-indigo-900">Advanced Mode Summary</h4>
                    <div className="text-sm text-indigo-700 mt-1 space-y-1">
                      <p><strong>Products:</strong> {(config.selectedProducts || []).length} selected</p>
                      <p><strong>Customers:</strong> {(config.selectedCustomers || []).length} selected</p>
                      <p><strong>Locations:</strong> {(config.selectedLocations || []).length} selected</p>
                      <p><strong>Precise Combinations:</strong> {
                        (config.selectedProducts || []).length * 
                        (config.selectedCustomers || []).length * 
                        (config.selectedLocations || []).length
                      }</p>
                      <p className="text-xs text-indigo-600 mt-2">
                        Each Product-Customer-Location combination will be forecasted separately without aggregation.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {!multiSelectMode && config.selectedProduct && config.selectedCustomer && config.selectedLocation && (
              <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-sm text-green-800">
                  <strong>Selected Combination:</strong> {config.selectedProduct} → {config.selectedCustomer} → {config.selectedLocation}
                </p>
                <p className="text-xs text-green-600 mt-1">
                  No aggregation will be applied - using exact data points for this combination
                </p>
              </div>
            )}
          </div>
        ) : simpleMultiSelect ? (
          /* Simple Multi-Select Mode */
          <div className="space-y-6">
            {/* Forecast By */}
            <div className='mt-6'>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Choose Forecast Dimension
              </label>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Product Option */}
                <div 
                  className={`relative p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                    config.forecastBy === 'product' 
                      ? 'border-blue-500 bg-blue-50 shadow-md' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => handleChange('forecastBy', 'product')}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${
                      config.forecastBy === 'product' ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-500'
                    }`}>
                      <Package className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Products</h3>
                      <p className="text-sm text-gray-500">{productOptions.length} available</p>
                    </div>
                  </div>
                  {config.forecastBy === 'product' && (
                    <div className="absolute top-2 right-2">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                    </div>
                  )}
                </div>

                {/* Customer Option */}
                <div 
                  className={`relative p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                    config.forecastBy === 'customer' 
                      ? 'border-green-500 bg-green-50 shadow-md' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => handleChange('forecastBy', 'customer')}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${
                      config.forecastBy === 'customer' ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-500'
                    }`}>
                      <Users className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Customers</h3>
                      <p className="text-sm text-gray-500">{customerOptions.length} available</p>
                    </div>
                  </div>
                  {config.forecastBy === 'customer' && (
                    <div className="absolute top-2 right-2">
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                  )}
                </div>

                {/* Location Option */}
                <div 
                  className={`relative p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                    config.forecastBy === 'location' 
                      ? 'border-purple-500 bg-purple-50 shadow-md' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => handleChange('forecastBy', 'location')}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${
                      config.forecastBy === 'location' ? 'bg-purple-100 text-purple-600' : 'bg-gray-100 text-gray-500'
                    }`}>
                      <MapPin className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Locations</h3>
                      <p className="text-sm text-gray-500">{locationOptions.length} available</p>
                    </div>
                  </div>
                  {config.forecastBy === 'location' && (
                    <div className="absolute top-2 right-2">
                      <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Multi-Select Items */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <div className="flex items-center space-x-2">
                  {getForecastTypeIcon()}
                  <span>Select Multiple {config.forecastBy.charAt(0).toUpperCase() + config.forecastBy.slice(1)}s</span>
                </div>
              </label>
              
              <div className="border border-gray-300 rounded-lg p-3 max-h-60 overflow-y-auto">
                <div className="space-y-2">
                  {getOptionsForType().map((option) => (
                    <label key={option} className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={(config.selectedItems || []).includes(option)}
                        onChange={() => handleSimpleMultiSelectChange(option)}
                        className="rounded border-gray-300 text-green-600 focus:ring-green-500"
                      />
                      <span className="text-sm text-gray-700">{option}</span>
                    </label>
                  ))}
                </div>
                
                {(config.selectedItems || []).length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <p className="text-sm font-medium text-green-900 mb-2">
                      Selected {config.forecastBy}s ({(config.selectedItems || []).length}):
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {(config.selectedItems || []).map((item) => (
                        <span
                          key={item}
                          className="inline-flex items-center px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full"
                        >
                          {item}
                          <button
                            type="button"
                            onClick={() => handleSimpleMultiSelectChange(item)}
                            className="ml-1 text-green-600 hover:text-green-800"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              
              {getOptionsForType().length === 0 && (
                <p className="mt-2 text-sm text-gray-500">
                  No {config.forecastBy}s available in data
                </p>
              )}
            </div>
          </div>
        ) : (
          /* Simple Single Dimension Selection */
          <div className="space-y-6">
            {/* Forecast By */}
            <div className='mt-6'>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Choose Forecast Dimension
              </label>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Product Option */}
                <div 
                  className={`relative p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                    config.forecastBy === 'product' 
                      ? 'border-blue-500 bg-blue-50 shadow-md' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => handleChange('forecastBy', 'product')}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${
                      config.forecastBy === 'product' ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-500'
                    }`}>
                      <Package className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Product</h3>
                      <p className="text-sm text-gray-500">{productOptions.length} available</p>
                    </div>
                  </div>
                  {config.forecastBy === 'product' && (
                    <div className="absolute top-2 right-2">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                    </div>
                  )}
                </div>

                {/* Customer Option */}
                <div 
                  className={`relative p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                    config.forecastBy === 'customer' 
                      ? 'border-green-500 bg-green-50 shadow-md' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => handleChange('forecastBy', 'customer')}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${
                      config.forecastBy === 'customer' ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-500'
                    }`}>
                      <Users className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Customer</h3>
                      <p className="text-sm text-gray-500">{customerOptions.length} available</p>
                    </div>
                  </div>
                  {config.forecastBy === 'customer' && (
                    <div className="absolute top-2 right-2">
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                  )}
                </div>

                {/* Location Option */}
                <div 
                  className={`relative p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                    config.forecastBy === 'location' 
                      ? 'border-purple-500 bg-purple-50 shadow-md' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => handleChange('forecastBy', 'location')}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${
                      config.forecastBy === 'location' ? 'bg-purple-100 text-purple-600' : 'bg-gray-100 text-gray-500'
                    }`}>
                      <MapPin className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Location</h3>
                      <p className="text-sm text-gray-500">{locationOptions.length} available</p>
                    </div>
                  </div>
                  {config.forecastBy === 'location' && (
                    <div className="absolute top-2 right-2">
                      <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Selected Item */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <div className="flex items-center space-x-2">
                  {getForecastTypeIcon()}
                  <span>Select {config.forecastBy.charAt(0).toUpperCase() + config.forecastBy.slice(1)}</span>
                </div>
              </label>
              <select
                value={config.selectedItem}
                onChange={(e) => handleChange('selectedItem', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors bg-white"
                disabled={getOptionsForType().length === 0}
              >
                <option value="">
                  {getOptionsForType().length === 0 
                    ? `No ${config.forecastBy}s available in data` 
                    : `Choose a ${config.forecastBy} (${getOptionsForType().length} available)`
                  }
                </option>
                {getOptionsForType().map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              {config.selectedItem && (
                <p className="mt-2 text-sm text-gray-600">
                  Selected: <span className="font-medium text-gray-900">{config.selectedItem}</span>
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Time Configuration */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center mb-6">
          <Clock className="w-6 h-6 text-green-600 mr-2" />
          <h2 className="text-xl font-semibold text-gray-900">Time Configuration</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Time Interval */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Calendar className="w-4 h-4 inline mr-1" />
              Time Interval
            </label>
            <select
              value={config.interval}
              onChange={(e) => handleChange('interval', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
            >
              <option value="week">Weekly</option>
              <option value="month">Monthly</option>
              <option value="year">Yearly</option>
            </select>
          </div>

          {/* Historic Period */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Historic Periods
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={config.historicPeriod}
              onChange={(e) => handleChange('historicPeriod', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
            />
            <p className="mt-1 text-xs text-gray-500">
              Number of past {config.interval}s to analyze
            </p>
          </div>

          {/* Forecast Period */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Forecast Periods
            </label>
            <input
              type="number"
              min="1"
              max="52"
              value={config.forecastPeriod}
              onChange={(e) => handleChange('forecastPeriod', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
            />
            <p className="mt-1 text-xs text-gray-500">
              Number of future {config.interval}s to forecast
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};