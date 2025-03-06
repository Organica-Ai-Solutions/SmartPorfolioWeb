import React from 'react';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

type AllocationChartProps = {
  allocations: Record<string, number>;
};

export function AllocationChart({ allocations }: AllocationChartProps) {
  // Generate random colors for the chart
  const generateColors = (count: number) => {
    const colors = [];
    for (let i = 0; i < count; i++) {
      const hue = (i * 137.5) % 360; // Use golden angle approximation for even distribution
      colors.push(`hsl(${hue}, 70%, 60%)`);
    }
    return colors;
  };

  // Prepare data for the chart
  const tickers = Object.keys(allocations || {});
  const values = tickers.map(ticker => allocations[ticker]);
  const backgroundColor = generateColors(tickers.length);
  
  const chartData = {
    labels: tickers,
    datasets: [
      {
        data: values,
        backgroundColor,
        borderColor: backgroundColor.map(color => color.replace('60%', '50%')),
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
          },
          padding: 15,
        },
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            const label = context.label || '';
            const value = context.raw || 0;
            return `${label}: ${(value * 100).toFixed(2)}%`;
          },
        },
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