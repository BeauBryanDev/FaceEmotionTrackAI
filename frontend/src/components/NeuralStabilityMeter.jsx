import React, { useMemo } from "react"

/**
 * NEURAL STABILITY METER
 * Uses entropy to estimate model stability
 */

const NeuralStabilityMeter = ({ entropy }) => {

  /**
   * Normalize entropy
   * max entropy for 8 classes ≈ log(8)
   */
  const normalized = useMemo(() => {
    if (entropy === undefined || entropy === null) return 0
    const maxEntropy = Math.log(8)
    return Math.min(entropy / maxEntropy, 1)
  }, [entropy])

  /**
   * Stability is inverse entropy
   */
  const stability = (1 - normalized) * 100

  /**
   * Status label
   */
  const status = useMemo(() => {
    if (stability > 75) return "STABLE"
    if (stability > 40) return "VARIABLE"
    return "UNSTABLE"
  }, [stability])

  return (
    <div className="bg-surface-1/40 border border-purple-800/40 p-4 relative overflow-hidden flex flex-col h-[160px]">

      {/* HEADER */}
      <div className="flex justify-between items-center mb-3">
        <span className="text-[10px] font-mono text-purple-400 tracking-widest">
          08 // NEURAL_STABILITY
        </span>

        <div className="flex gap-1">
          {[1,2,3].map(i => (
            <div
              key={i}
              className="w-1 h-1 bg-neon-purple animate-pulse"
              style={{animationDelay:`${i*0.2}s`}}
            />
          ))}
        </div>
      </div>

      {/* MAIN METER */}
      <div className="flex flex-col flex-1 justify-center gap-3">

        {/* BAR */}
        <div className="relative h-3 bg-purple-900 overflow-hidden border border-purple-800">

          <div
            className="h-full bg-neon-purple shadow-neon-sm transition-all duration-700"
            style={{ width: `${stability}%` }}
          />

          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-pulse opacity-20"/>
        </div>

        {/* VALUES */}
        <div className="flex justify-between items-center text-[10px] font-mono">

          <span className="text-purple-400">
            STABILITY
          </span>

          <span className="text-neon-purple font-bold">
            {stability.toFixed(1)}%
          </span>

        </div>

        {/* STATUS */}
        <div className="flex justify-between text-[9px] font-mono text-purple-500">

          <span>
            ENTROPY: {entropy ? entropy.toFixed(3) : "0.000"}
          </span>

          <span className="text-neon-purple">
            {status}
          </span>

        </div>

      </div>

      {/* HUD DECOR */}
      <div className="absolute top-0 right-0 w-16 h-[1px] bg-neon-purple/20 rotate-45 translate-x-4 -translate-y-4"/>

    </div>
  )
}

export default NeuralStabilityMeter
