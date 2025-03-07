import React, { useEffect } from 'react';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

type AllocationChartProps = {
  allocations: Record<string, number>;
};

export function AllocationChart({ allocations }: AllocationChartProps) {
  // Generate vibrant colors for the chart
  const generateColors = (count: number) => {
    const baseColors = [
      'hsl(348, 83%, 47%)',   // Red
      'hsl(120, 73%, 45%)',   // Green
      'hsl(260, 73%, 55%)',   // Purple
      'hsl(217, 85%, 60%)',   // Blue
      'hsl(39, 100%, 50%)',   // Yellow
      'hsl(180, 73%, 45%)',   // Teal
      'hsl(288, 59%, 58%)',   // Pink
      'hsl(15, 73%, 55%)',    // Orange
    ];
    
    const colors = [];
    for (let i = 0; i < count; i++) {
      colors.push(baseColors[i % baseColors.length]);
    }
    return colors;
  };

  // Convert allocations to percentage values (0-100)
  const convertToPercentages = (allocData: Record<string, number>) => {
    // Convert directly to percentage (multiply by 100)
    return Object.entries(allocData).map(([ticker, value]) => ({
      ticker,
      percentage: value * 100
    }));
  };

  useEffect(() => {
    // Log allocations for debugging
    console.log("Raw allocations:", allocations);
    if (allocations) {
      const percentages = convertToPercentages(allocations);
      console.log("Percentages:", percentages);
    }
  }, [allocations]);

  // Prepare data for the chart
  const tickers = Object.keys(allocations || {});
  const values = Object.values(allocations || {}).map(val => val * 100); // Direct percentages
  const backgroundColor = generateColors(tickers.length);
  
  const chartData = {
    labels: tickers,
    datasets: [
      {
        data: values,
        backgroundColor,
        borderColor: backgroundColor.map(color => color.replace(')', ', 0.8)')),
        borderWidth: 2,
      },
    ],
  };

  // Display percentages directly on the chart
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right' as const,
        labels: {
          font: {
            size: 14,
            weight: 'bold' as const,
          },
          padding: 15,
          color: 'white',
          usePointStyle: true,
          pointStyle: 'circle',
        },
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            const label = context.label || '';
            const value = context.raw || 0;
            return `${label}: ${value.toFixed(1)}%`;
          },
        },
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        titleFont: {
          size: 14,
          weight: 'bold' as const,
        },
        bodyFont: {
          size: 13,
        },
        displayColors: true,
        padding: 10,
        cornerRadius: 4,
      },
    },
    // Add these settings to emphasize the sections
    cutout: '0%',  // No cutout (full pie)
    radius: '90%', // Larger radius
    animation: {
      animateRotate: true,
      animateScale: true
    },
  };

  if (!tickers.length) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        No allocation data available
      </div>
    );
  }

  return (
    <div className="h-64">
      <Pie data={chartData} options={options} />
    </div>
  );
} 