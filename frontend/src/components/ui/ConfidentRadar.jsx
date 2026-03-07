import React, { useMemo } from "react";

/**
 * CONFIDENCE RADAR
 * Visualizes model uncertainty using entropy.
 * Low entropy  -> high confidence
 * High entropy -> low confidence
 */

const ConfidenceRadar = ({ entropy }) => {

  /**
   * Entropy normalization
   * max entropy for 8 classes ≈ log(8) = 2.079
   */
  const normalized = useMemo(() => {
    if (entropy === undefined || entropy === null) return 0
    const maxEntropy = Math.log(8)
    return Math.min(entropy / maxEntropy, 1)
  }, [entropy])

  /**
   * Convert entropy → ring radius
   */
  const radius = 20 + normalized * 70

  /**
   * Confidence value (inverse entropy)
   */
  const confidence = ((1 - normalized) * 100).toFixed(1)

  return (
    <div className="relative w-full h-full min-h-[260px] flex items-center justify-center bg-surface-1/40 border border-purple-900/40 overflow-hidden group">

      {/* CORNER HUD ACCENTS */}
      <div className="absolute top-0 left-0 w-8 h-8 border-t border-l border-purple-500/30" />
      <div className="absolute top-0 right-0 w-8 h-8 border-t border-r border-purple-500/30" />
      <div className="absolute bottom-0 left-0 w-8 h-8 border-b border-l border-purple-500/30" />
      <div className="absolute bottom-0 right-0 w-8 h-8 border-b border-r border-purple-500/30" />

      {/* MODULE HEADER */}
      <div className="absolute top-3 left-4 flex flex-col">
        <span className="text-purple-400 font-mono text-[9px] tracking-[0.3em] font-bold">
          NEURALCORE.CONFIDENCE
        </span>
        <span className="text-purple-600 font-mono text-[7px]">
          SIGNAL_STABILITY_INDEX
        </span>
      </div>

      {/* SVG RADAR */}
      <svg width="220" height="220" className="overflow-visible">

        {/* GRID RINGS */}
        <circle cx="110" cy="110" r="90" stroke="#6b21a8" strokeOpacity="0.2" fill="none"/>
        <circle cx="110" cy="110" r="65" stroke="#6b21a8" strokeOpacity="0.2" fill="none"/>
        <circle cx="110" cy="110" r="40" stroke="#6b21a8" strokeOpacity="0.2" fill="none"/>

        {/* CROSSHAIR */}
        <line x1="20" y1="110" x2="200" y2="110" stroke="#7c3aed" strokeOpacity="0.2"/>
        <line x1="110" y1="20" x2="110" y2="200" stroke="#7c3aed" strokeOpacity="0.2"/>

        {/* ENTROPY RING */}
        <circle
          cx="110"
          cy="110"
          r={radius}
          stroke="#bf00ff"
          strokeWidth="2"
          fill="none"
          strokeOpacity="0.85"
          style={{
            filter: "drop-shadow(0 0 8px rgba(191,0,255,0.9))"
          }}
        />

        {/* CENTER CORE */}
        <circle
          cx="110"
          cy="110"
          r="6"
          fill="#bf00ff"
          style={{
            filter: "drop-shadow(0 0 8px rgba(191,0,255,0.9))"
          }}
        />

        <circle
          cx="110"
          cy="110"
          r="12"
          stroke="#bf00ff"
          strokeOpacity="0.25"
          fill="none"
        />

      </svg>

      {/* CONFIDENCE TEXT */}
      <div className="absolute bottom-6 text-center font-mono">
        <div className="text-[10px] text-purple-400 tracking-widest">
          MODEL_CONFIDENCE
        </div>
        <div className="text-neon-purple text-xl font-bold">
          {confidence}%
        </div>
      </div>

      {/* ENTROPY VALUE */}
      <div className="absolute bottom-2 right-3 text-[9px] text-purple-500 font-mono">
        ENTROPY: {entropy ? entropy.toFixed(3) : "0.000"}
      </div>

    </div>
  )
}

export default ConfidenceRadar
