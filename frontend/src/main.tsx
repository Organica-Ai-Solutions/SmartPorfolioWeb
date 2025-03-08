import React from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// Add a global error handler to intercept the specific error
const originalError = console.error;
console.error = function(...args) {
  // Check if this is the error we're trying to catch
  const errorStr = args.join(' ');
  if (errorStr.includes('Invalid response data')) {
    console.log('⚠️ Caught Invalid response data error - activating global failsafe!');
    
    // Store flag in session storage to trigger global failsafe mode
    sessionStorage.setItem('USE_GLOBAL_FAILSAFE', 'true');
    
    // Reload the page to reset everything and use the failsafe
    window.location.reload();
    
    return; // Don't pass this error to the original handler
  }
  
  // Pass all other errors to the original handler
  originalError.apply(console, args);
};

// Add a version check message to know we're running the latest code
console.log('▶️ Running SmartPortfolio v1.2.0-failsafe');

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
