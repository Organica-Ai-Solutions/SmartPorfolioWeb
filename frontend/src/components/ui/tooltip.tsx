import * as React from "react"
import * as TooltipPrimitive from "@radix-ui/react-tooltip"
import { cn } from "../../lib/utils"

const TooltipProvider = TooltipPrimitive.Provider

const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Content
    ref={ref}
    sideOffset={sideOffset}
    className={cn(
      "z-50 overflow-hidden rounded-lg border border-white/10 bg-black/95 px-4 py-3",
      "text-sm text-white shadow-xl backdrop-blur-sm max-w-[300px]",
      "animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95",
      "data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2",
      "data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
      className
    )}
    {...props}
  />
))

TooltipContent.displayName = TooltipPrimitive.Content.displayName

interface TooltipProps {
  content: React.ReactNode | {
    title?: string;
    description?: string;
    value?: string | number;
    trend?: 'up' | 'down' | 'neutral';
    metrics?: Array<{
      label: string;
      value: string | number;
      trend?: 'up' | 'down' | 'neutral';
    }>;
  };
  children: React.ReactNode;
  side?: "top" | "right" | "bottom" | "left";
  sideOffset?: number;
  align?: "start" | "center" | "end";
  delayDuration?: number;
  className?: string;
}

const Tooltip = React.memo(({
  content,
  children,
  side = "top",
  sideOffset = 4,
  align = "center",
  delayDuration = 200,
  className,
}: TooltipProps) => {
  const renderContent = () => {
    if (typeof content === 'string' || React.isValidElement(content)) {
      return content;
    }

    const tooltipContent = content as Exclude<typeof content, React.ReactNode>;
    
    return (
      <div className="space-y-2">
        {tooltipContent.title && (
          <div className="font-medium text-purple-400">{tooltipContent.title}</div>
        )}
        {tooltipContent.description && (
          <div className="text-sm text-gray-300">{tooltipContent.description}</div>
        )}
        {tooltipContent.value && (
          <div className={cn(
            "text-lg font-medium",
            tooltipContent.trend === 'up' && "text-green-400",
            tooltipContent.trend === 'down' && "text-red-400"
          )}>
            {tooltipContent.value}
          </div>
        )}
        {tooltipContent.metrics && tooltipContent.metrics.length > 0 && (
          <div className="grid gap-2 pt-2 border-t border-white/10">
            {tooltipContent.metrics.map((metric, index) => (
              <div key={index} className="flex justify-between items-center">
                <span className="text-sm text-gray-400">{metric.label}</span>
                <span className={cn(
                  "font-medium",
                  metric.trend === 'up' && "text-green-400",
                  metric.trend === 'down' && "text-red-400"
                )}>
                  {metric.value}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <TooltipPrimitive.Root delayDuration={delayDuration}>
      <TooltipPrimitive.Trigger asChild>
        {children}
      </TooltipPrimitive.Trigger>
      <TooltipContent 
        side={side} 
        sideOffset={sideOffset} 
        align={align}
        className={className}
      >
        {renderContent()}
      </TooltipContent>
    </TooltipPrimitive.Root>
  );
});

Tooltip.displayName = "Tooltip"

export {
  Tooltip,
  TooltipProvider,
} 