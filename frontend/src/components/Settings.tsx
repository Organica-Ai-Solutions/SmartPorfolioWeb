import { useState, useEffect } from 'react';
import { Dialog } from '@headlessui/react';
import { XMarkIcon } from '@heroicons/react/24/outline';

interface SettingsProps {
  isOpen: boolean;
  onClose: () => void;
}

export interface AlpacaSettings {
  apiKey: string;
  secretKey: string;
  isPaper: boolean;
}

export function Settings({ isOpen, onClose }: SettingsProps) {
  const [apiKey, setApiKey] = useState('');
  const [secretKey, setSecretKey] = useState('');
  const [isPaper, setIsPaper] = useState(true);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // Load saved settings on component mount
  useEffect(() => {
    const savedSettings = localStorage.getItem('alpacaSettings');
    if (savedSettings) {
      const settings = JSON.parse(savedSettings) as AlpacaSettings;
      setApiKey(settings.apiKey || '');
      setSecretKey(settings.secretKey || '');
      setIsPaper(settings.isPaper !== false); // Default to true if not specified
    }
  }, [isOpen]);

  const handleSave = () => {
    const settings: AlpacaSettings = {
      apiKey,
      secretKey,
      isPaper
    };
    
    localStorage.setItem('alpacaSettings', JSON.stringify(settings));
    setSaveSuccess(true);
    
    // Reset success message after 3 seconds
    setTimeout(() => {
      setSaveSuccess(false);
    }, 3000);
  };

  return (
    <Dialog
      open={isOpen}
      onClose={onClose}
      className="relative z-50"
    >
      <div className="fixed inset-0 bg-black/30 backdrop-blur-sm" aria-hidden="true" />
      
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <Dialog.Panel className="w-full max-w-md rounded-2xl bg-white dark:bg-gray-800 p-6 shadow-xl">
          <div className="flex justify-between items-center mb-4">
            <Dialog.Title className="text-xl font-semibold text-gray-900 dark:text-white">
              API Settings
            </Dialog.Title>
            <button 
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Alpaca API Key
              </label>
              <input
                type="text"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
                         shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 
                         dark:bg-gray-700 dark:text-white"
                placeholder="Enter your Alpaca API key"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Alpaca Secret Key
              </label>
              <input
                type="password"
                value={secretKey}
                onChange={(e) => setSecretKey(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
                         shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 
                         dark:bg-gray-700 dark:text-white"
                placeholder="Enter your Alpaca Secret key"
              />
            </div>
            
            <div className="flex items-center">
              <input
                id="paper-trading"
                type="checkbox"
                checked={isPaper}
                onChange={(e) => setIsPaper(e.target.checked)}
                className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 
                         border-gray-300 rounded"
              />
              <label htmlFor="paper-trading" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                Use Paper Trading (recommended)
              </label>
            </div>
            
            {saveSuccess && (
              <div className="text-sm text-green-600 dark:text-green-400">
                Settings saved successfully!
              </div>
            )}
            
            <div className="flex justify-end mt-6">
              <button
                onClick={handleSave}
                className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white 
                         rounded-md hover:opacity-90 transition-opacity"
              >
                Save Settings
              </button>
            </div>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}

export default Settings; 