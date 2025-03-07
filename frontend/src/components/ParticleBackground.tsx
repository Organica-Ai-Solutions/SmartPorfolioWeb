import React from 'react';
import { SparklesCore } from './ui/sparkles';

export const ParticleBackground: React.FC = () => {
  return (
    <SparklesCore
      id="tsparticlesfullpage"
      background="transparent"
      minSize={0.8}
      maxSize={1.6}
      particleDensity={100}
      className="w-full h-full"
      particleColor="#FFFFFF"
      speed={1}
    />
  );
};

export default ParticleBackground; 