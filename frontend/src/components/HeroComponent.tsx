import React from 'react';

export function HeroComponent() {
  return (
    <div className="flex flex-col items-center justify-center py-20 w-full bg-gradient-to-br from-gray-900 via-black to-gray-900">
      <div className="relative group">
        <div className="absolute -inset-0.5 bg-gradient-to-r from-pink-600 to-purple-600 rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-1000 group-hover:duration-200 animate-tilt"></div>
        <div className="relative px-7 py-6 bg-black rounded-lg leading-none">
          <div className="flex flex-col items-center space-y-2">
            <div className="flex items-center">
              <h1 className="text-6xl font-extrabold tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-pink-500 via-blue-500 to-purple-500 animate-gradient-x">
                Smart Portfolio 
              </h1>
              <span className="ml-3 inline-flex items-center px-3 py-0.5 rounded-full text-xs font-semibold leading-5 uppercase tracking-wide bg-gradient-to-r from-indigo-500 to-purple-600 text-white">
                PRO
              </span>
            </div>
            <p className="text-slate-300 font-medium tracking-wide text-lg">
              Intelligent investing powered by AI
            </p>
          </div>
        </div>
      </div>
    </div>
  );
} 