import React, { useState, useRef, useEffect } from 'react';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  PointElement, 
  Title, 
  Tooltip, 
  Legend,
  ChartOptions
} from 'chart.js';
import { Scatter } from 'react-chartjs-2';
import { Slider } from './ui/slider';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  Title,
  Tooltip,
  Legend
);

interface CorrelationHeatmapProps {
  correlationMatrix: number[][];
  tickers: string[];
  onTickerSelect?: (ticker: string) => void;
}

export function CorrelationHeatmap({ correlationMatrix, tickers, onTickerSelect }: CorrelationHeatmapProps) {
  const [highlightedTicker, setHighlightedTicker] = useState<string | null>(null);
  const [correlationThreshold, setCorrelationThreshold] = useState<number[]>([0.5]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<any>(null);

  // Generate color based on correlation value
  const getCorrelationColor = (value: number): string => {
    // Strong negative correlation: red
    // No correlation: white/gray
    // Strong positive correlation: green
    
    if (value >= 0) {
      // Positive correlation: green gradient
      const intensity = Math.min(Math.round(value * 255), 255);
      return `rgba(16, 185, 129, ${value.toFixed(2)})`;
    } else {
      // Negative correlation: red gradient
      const intensity = Math.min(Math.round(Math.abs(value) * 255), 255);
      return `rgba(239, 68, 68, ${Math.abs(value).toFixed(2)})`;
    }
  };

  // Format data for heatmap visualization
  const formatHeatmapData = () => {
    const threshold = correlationThreshold[0];
    
    // Create data points for the heatmap
    const dataPoints = [];
    
    for (let i = 0; i < tickers.length; i++) {
      for (let j = 0; j < tickers.length; j++) {
        const correlation = correlationMatrix[i][j];
        
        // Skip self-correlations (always 1.0)
        if (i === j) continue;
        
        // Apply threshold filter
        if (Math.abs(correlation) < threshold) continue;
        
        // Highlight selected ticker if any
        const isHighlighted = highlightedTicker && 
          (tickers[i] === highlightedTicker || tickers[j] === highlightedTicker);
        
        dataPoints.push({
          x: i,
          y: j,
          correlation,
          color: getCorrelationColor(correlation),
          opacity: isHighlighted || !highlightedTicker ? 1 : 0.3,
          ticker1: tickers[i],
          ticker2: tickers[j]
        });
      }
    }
    
    return {
      datasets: [
        {
          label: 'Asset Correlations',
          data: dataPoints,
          backgroundColor: dataPoints.map(point => point.color),
          pointRadius: 15,
          pointHoverRadius: 18,
          pointStyle: 'rectRot',
          hoverBackgroundColor: dataPoints.map(point => point.color.replace(')', ', 1)')),
        }
      ]
    };
  };

  // Custom drawing function for the heatmap
  const drawHeatmap = (chart: any) => {
    const { ctx, data, chartArea, scales } = chart;
    
    if (!chartArea || !data || !data.datasets || data.datasets.length === 0) return;
    
    const dataset = data.datasets[0];
    const points = dataset.data;
    
    // Clear the canvas
    ctx.save();
    ctx.clearRect(0, 0, chart.width, chart.height);
    
    // Draw the grid
    ctx.fillStyle = 'rgba(17, 24, 39, 1)';
    ctx.fillRect(chartArea.left, chartArea.top, chartArea.width, chartArea.height);
    
    // Draw the heatmap cells
    points.forEach((point: any) => {
      const x = scales.x.getPixelForValue(point.x);
      const y = scales.y.getPixelForValue(point.y);
      const size = 25; // Cell size
      
      // Draw cell
      ctx.globalAlpha = point.opacity;
      ctx.fillStyle = point.color;
      ctx.fillRect(x - size/2, y - size/2, size, size);
      
      // Draw correlation value
      ctx.fillStyle = 'white';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(point.correlation.toFixed(2), x, y);
    });
    
    // Draw ticker labels
    ctx.globalAlpha = 1;
    ctx.fillStyle = 'white';
    ctx.font = '12px Arial';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    
    // X-axis labels (vertical)
    tickers.forEach((ticker, i) => {
      const x = scales.x.getPixelForValue(i);
      const isHighlighted = ticker === highlightedTicker;
      
      ctx.save();
      ctx.translate(x, chartArea.bottom + 10);
      ctx.rotate(-Math.PI / 2);
      
      if (isHighlighted) {
        ctx.font = 'bold 12px Arial';
        ctx.fillStyle = '#8b5cf6';
      } else {
        ctx.font = '12px Arial';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      }
      
      ctx.fillText(ticker, 0, 0);
      ctx.restore();
    });
    
    // Y-axis labels
    tickers.forEach((ticker, i) => {
      const y = scales.y.getPixelForValue(i);
      const isHighlighted = ticker === highlightedTicker;
      
      if (isHighlighted) {
        ctx.font = 'bold 12px Arial';
        ctx.fillStyle = '#8b5cf6';
      } else {
        ctx.font = '12px Arial';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      }
      
      ctx.fillText(ticker, chartArea.left - 10, y);
    });
    
    ctx.restore();
  };

  // Chart options
  const options: ChartOptions<'scatter'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 500
    },
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const point = context.raw as any;
            return [
              `${point.ticker1} ↔ ${point.ticker2}`,
              `Correlation: ${point.correlation.toFixed(2)}`
            ];
          }
        },
        backgroundColor: 'rgba(17, 24, 39, 0.9)',
        titleColor: 'rgba(255, 255, 255, 0.9)',
        bodyColor: 'rgba(255, 255, 255, 0.8)',
        borderColor: 'rgba(63, 63, 70, 1)',
        borderWidth: 1,
        padding: 12,
      }
    },
    scales: {
      x: {
        type: 'linear',
        min: -0.5,
        max: tickers.length - 0.5,
        grid: {
          display: false
        },
        ticks: {
          display: false
        },
        border: {
          display: false
        }
      },
      y: {
        type: 'linear',
        min: -0.5,
        max: tickers.length - 0.5,
        grid: {
          display: false
        },
        ticks: {
          display: false
        },
        border: {
          display: false
        }
      }
    },
    onClick: (event, elements) => {
      if (elements && elements.length > 0 && onTickerSelect) {
        const index = elements[0].index;
        const point = chartRef.current?.data?.datasets[0]?.data[index];
        
        if (point) {
          // Toggle between the two tickers on click
          const ticker = highlightedTicker === point.ticker1 ? point.ticker2 : point.ticker1;
          setHighlightedTicker(ticker);
          onTickerSelect(ticker);
        }
      }
    }
  };

  // Handle threshold change
  const handleThresholdChange = (values: number[]) => {
    setCorrelationThreshold(values);
  };

  // Get correlation statistics
  const getCorrelationStats = () => {
    let highestPositive = { value: -1, pair: ['', ''] };
    let highestNegative = { value: 1, pair: ['', ''] };
    let totalPositive = 0;
    let totalNegative = 0;
    let countPositive = 0;
    let countNegative = 0;
    
    for (let i = 0; i < tickers.length; i++) {
      for (let j = i + 1; j < tickers.length; j++) {
        const correlation = correlationMatrix[i][j];
        
        if (correlation > 0) {
          totalPositive += correlation;
          countPositive++;
          
          if (correlation > highestPositive.value) {
            highestPositive = { value: correlation, pair: [tickers[i], tickers[j]] };
          }
        } else if (correlation < 0) {
          totalNegative += correlation;
          countNegative++;
          
          if (correlation < highestNegative.value) {
            highestNegative = { value: correlation, pair: [tickers[i], tickers[j]] };
          }
        }
      }
    }
    
    const avgPositive = countPositive > 0 ? totalPositive / countPositive : 0;
    const avgNegative = countNegative > 0 ? totalNegative / countNegative : 0;
    
    return {
      highestPositive,
      highestNegative,
      avgPositive,
      avgNegative,
      diversificationScore: Math.min(100, Math.round((1 - avgPositive) * 100))
    };
  };

  const stats = getCorrelationStats();

  return (
    <div className="flex flex-col w-full space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-white">Asset Correlation Heatmap</h3>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-400">Correlation Threshold:</span>
            <div className="w-32">
              <Slider
                value={correlationThreshold}
                min={0}
                max={1}
                step={0.05}
                onValueChange={handleThresholdChange}
              />
            </div>
            <span className="text-xs text-white">{correlationThreshold[0].toFixed(2)}</span>
          </div>
          
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-green-500 mr-1"></div>
            <span className="text-xs text-gray-400">Positive</span>
            <div className="w-3 h-3 rounded-full bg-red-500 mr-1 ml-2"></div>
            <span className="text-xs text-gray-400">Negative</span>
          </div>
        </div>
      </div>
      
      <div className="bg-gray-800/30 border border-gray-700 rounded-lg p-4">
        <div className="h-[450px]">
          <Scatter 
            ref={chartRef}
            data={formatHeatmapData()} 
            options={options}
            plugins={[
              {
                id: 'customHeatmap',
                beforeDraw: drawHeatmap
              }
            ]}
          />
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 p-3 rounded-lg border border-gray-700">
          <div className="text-xs text-gray-400 mb-1">Highest Positive Correlation</div>
          <div className="text-lg font-bold text-green-400">
            {stats.highestPositive.value.toFixed(2)}
          </div>
          <div className="text-xs text-gray-300">
            {stats.highestPositive.pair[0]} ↔ {stats.highestPositive.pair[1]}
          </div>
        </div>
        
        <div className="bg-gray-800 p-3 rounded-lg border border-gray-700">
          <div className="text-xs text-gray-400 mb-1">Highest Negative Correlation</div>
          <div className="text-lg font-bold text-red-400">
            {stats.highestNegative.value.toFixed(2)}
          </div>
          <div className="text-xs text-gray-300">
            {stats.highestNegative.pair[0]} ↔ {stats.highestNegative.pair[1]}
          </div>
        </div>
        
        <div className="bg-gray-800 p-3 rounded-lg border border-gray-700">
          <div className="text-xs text-gray-400 mb-1">Average Correlation</div>
          <div className="text-lg font-bold text-blue-400">
            {((stats.avgPositive * countPositive + stats.avgNegative * countNegative) / 
              (countPositive + countNegative)).toFixed(2)}
          </div>
          <div className="text-xs text-gray-300">
            Between all assets
          </div>
        </div>
        
        <div className="bg-gray-800 p-3 rounded-lg border border-gray-700">
          <div className="text-xs text-gray-400 mb-1">Diversification Score</div>
          <div className="text-lg font-bold text-purple-400">
            {stats.diversificationScore}/100
          </div>
          <div className="text-xs text-gray-300">
            Based on correlation patterns
          </div>
        </div>
      </div>
      
      <div className="text-xs text-gray-400 text-center">
        This heatmap shows the correlation between assets in your portfolio. 
        Lower correlation between assets indicates better diversification.
        Click on any cell to highlight correlations for that asset.
      </div>
    </div>
  );
} 