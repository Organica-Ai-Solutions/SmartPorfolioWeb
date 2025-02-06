import { Button } from "./ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import { PortfolioAnalysis } from "../types/portfolio";

interface PortfolioExplanationProps {
  analysis: PortfolioAnalysis;
}

export function PortfolioExplanation({ analysis }: PortfolioExplanationProps) {
  const explanations = analysis.ai_insights?.explanations;

  // Return null if no AI insights are available
  if (!analysis.ai_insights || !explanations) return null;

  // Ensure all required sections exist
  const sections = {
    summary: explanations.summary || { en: 'No summary available.', es: 'No hay resumen disponible.' },
    risk_analysis: explanations.risk_analysis || { en: 'No risk analysis available.', es: 'No hay análisis de riesgo disponible.' },
    diversification_analysis: explanations.diversification_analysis || { en: 'No diversification analysis available.', es: 'No hay análisis de diversificación disponible.' },
    market_context: explanations.market_context || { en: 'No market context available.', es: 'No hay contexto de mercado disponible.' },
    stress_test_interpretation: explanations.stress_test_interpretation || { en: 'No stress test analysis available.', es: 'No hay análisis de prueba de estrés disponible.' }
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" className="w-full md:w-auto">
          <span className="mr-2">✨</span> AI Explanation
        </Button>
      </DialogTrigger>
      <DialogContent className="flex flex-col gap-0 p-0 sm:max-h-[min(640px,80vh)] sm:max-w-[800px]">
        <div className="overflow-y-auto">
          <DialogHeader className="px-6 pt-6">
            <DialogTitle className="text-xl bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
              Portfolio Analysis Explanation
            </DialogTitle>
            <DialogDescription className="text-base text-gray-300">
              Detailed insights and recommendations for your portfolio
            </DialogDescription>
          </DialogHeader>

          <div className="p-6 space-y-6">
            {/* Summary Section */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-purple-400">Summary</h3>
              <div className="bg-white/5 rounded-lg p-4 space-y-2">
                <p className="text-white">{sections.summary.en}</p>
                <p className="text-gray-400 italic">{sections.summary.es}</p>
              </div>
            </div>

            {/* Risk Analysis Section */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-purple-400">Risk Analysis</h3>
              <div className="bg-white/5 rounded-lg p-4 space-y-2">
                <p className="text-white">{sections.risk_analysis.en}</p>
                <p className="text-gray-400 italic">{sections.risk_analysis.es}</p>
              </div>
            </div>

            {/* Diversification Analysis Section */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-purple-400">Diversification Analysis</h3>
              <div className="bg-white/5 rounded-lg p-4 space-y-2">
                <p className="text-white">{sections.diversification_analysis.en}</p>
                <p className="text-gray-400 italic">{sections.diversification_analysis.es}</p>
              </div>
            </div>

            {/* Market Context Section */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-purple-400">Market Context</h3>
              <div className="bg-white/5 rounded-lg p-4 space-y-2">
                <p className="text-white">{sections.market_context.en}</p>
                <p className="text-gray-400 italic">{sections.market_context.es}</p>
              </div>
            </div>

            {/* Stress Test Interpretation Section */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-purple-400">Stress Test Analysis</h3>
              <div className="bg-white/5 rounded-lg p-4 space-y-2">
                <p className="text-white">{sections.stress_test_interpretation.en}</p>
                <p className="text-gray-400 italic">{sections.stress_test_interpretation.es}</p>
              </div>
            </div>
          </div>
        </div>

        <DialogFooter className="border-t border-white/10 px-6 py-4">
          <DialogClose asChild>
            <Button type="button" variant="outline">Close</Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
} 