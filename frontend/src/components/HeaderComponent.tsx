import React from 'react';
import { Cog6ToothIcon } from '@heroicons/react/24/outline';

type HeaderComponentProps = {
  onSettingsClick?: () => void;
};

export function HeaderComponent({ onSettingsClick }: HeaderComponentProps = {}) {
  return (
    <header className="bg-black/60 backdrop-blur-md py-4 px-4 sm:px-6 lg:px-8 flex justify-between items-center border-b border-white/10">
      <div className="flex items-center space-x-2">
        {/* Logo icon */}
        <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-md flex items-center justify-center">
          <span className="text-white font-bold">SP</span>
        </div>
      </div>
      
      {/* Settings button */}
      {onSettingsClick && (
        <button
          onClick={onSettingsClick}
          className="p-2 rounded-full hover:bg-gray-700 transition-colors"
          aria-label="Settings"
        >
          <Cog6ToothIcon className="h-6 w-6 text-gray-300" />
        </button>
      )}
    </header>
  );
} 