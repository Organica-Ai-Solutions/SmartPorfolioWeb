import React from 'react';
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
      'hsl(120, 73%, 35%)',   // Green
      'hsl(260, 73%, 45%)',   // Purple
      'hsl(217, 85%, 55%)',   // Blue
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

  // Format percentage values for display
  const formatPercentages = (values: number[]) => {
    return values.map(value => value * 100);
  };

  // Prepare data for the chart
  const tickers = Object.keys(allocations || {});
  const values = tickers.map(ticker => allocations[ticker]);
  const backgroundColor = generateColors(tickers.length);
  const percentageValues = formatPercentages(values);
  
  const chartData = {
    labels: tickers,
    datasets: [
      {
        data: percentageValues,
        backgroundColor,
        borderColor: backgroundColor.map(color => color.replace(')', ', 0.8)')),
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right' as const,
        labels: {
          font: {
            size: 12,
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
        backgroundColor: 'rgba(20, 20, 20, 0.9)',
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