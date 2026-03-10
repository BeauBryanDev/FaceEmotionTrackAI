import React, { useMemo } from 'react';

const EmotionVectorField = ({ dynamics = [] }) => {
  if (!dynamics || dynamics.length === 0) return (
    <div className="h-64 flex items-center justify-center border border-purple-900 bg-purple-950/20 font-mono text-purple-700 text-[10px] tracking-widest">
      AWAITING_NEURAL_TELEMETRY...
    </div>
  );

  const last = dynamics[dynamics.length - 1];
  const size = 300;
  const padding = 40;
  const center = size / 2;
  const scale = (size - padding * 2) / 2; // Maps -1..1 to the grid

  // Map coordinates to SVG space
  const toSVG = (val, type) => {
    if (type === 'x') return center + val * scale;
    if (type === 'y') return center - val * scale; // Y is inverted in SVG
    return 0;
  };

  const dx = Math.cos(last.direction) * last.velocity;
  const dy = Math.sin(last.direction) * last.velocity;

  // Bearing in degrees (0 = North/Up)
  const bearing = ((last.direction * 180 / Math.PI + 360 + 90) % 360).toFixed(1);

  return (
    <div className="relative bg-black border border-purple-800 p-1 group">
      {/* HUD Background Decorations */}
      <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-neon-purple/50" />
      <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-neon-purple/50" />
      <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-neon-purple/50" />
      <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-neon-purple/50" />

      {/* Grid Overlay Text */}
      <div className="absolute top-2 left-6 right-6 flex justify-between font-mono text-[7px] text-purple-500 uppercase tracking-tighter opacity-70">
        <span>Sector_Alpha // Latency: 12ms</span>
        <span>Arousal_Vector_Field</span>
      </div>

      <svg viewBox={`0 0 ${size} ${size}`} className="w-full h-auto">
        <defs>
          <filter id="heatmapBlur" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>

          <radialGradient id="vectorGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#bf00ff" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#bf00ff" stopOpacity="0" />
          </radialGradient>

          <filter id="textGlow">
            <feGaussianBlur stdDeviation="1.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Heatmap Layer (Historical Density) */}
        <g filter="url(#heatmapBlur)">
          {dynamics.slice(-40).map((p, i) => (
            <circle
              key={i}
              cx={toSVG(p.valence, 'x')}
              cy={toSVG(p.arousal, 'y')}
              r={4}
              fill="#cc44ff"
              fillOpacity={0.1 + (i / 40) * 0.3}
            />
          ))}
        </g>

        {/* Tactical Grid */}
        <circle cx={center} cy={center} r={scale * 0.33} fill="none" stroke="#6b21a8" strokeOpacity="0.2" strokeDasharray="2,4" />
        <circle cx={center} cy={center} r={scale * 0.66} fill="none" stroke="#6b21a8" strokeOpacity="0.2" strokeDasharray="2,4" />
        <circle cx={center} cy={center} r={scale} fill="none" stroke="#6b21a8" strokeOpacity="0.4" strokeWidth="1" />

        <line x1={center} y1={padding} x2={center} y2={size - padding} stroke="#6b21a8" strokeOpacity="0.3" strokeWidth="0.5" />
        <line x1={padding} y1={center} x2={size - padding} y2={center} stroke="#6b21a8" strokeOpacity="0.3" strokeWidth="0.5" />

        {/* Quadrant Labels */}
        <g className="font-mono text-[7px] fill-purple-700 tracking-widest uppercase opacity-40">
          <text x={center + scale * 0.5} y={center - scale * 0.5}>STRIKE</text>
          <text x={center - scale * 0.85} y={center - scale * 0.5}>ALARM</text>
          <text x={center - scale * 0.85} y={center + scale * 0.5}>DORMANCY</text>
          <text x={center + scale * 0.5} y={center + scale * 0.5}>RECOVERY</text>
        </g>

        {/* Direction Indicator */}
        <line
          x1={toSVG(last.valence, 'x')}
          y1={toSVG(last.arousal, 'y')}
          x2={toSVG(last.valence + dx * 10, 'x')}
          y2={toSVG(last.arousal + dy * 10, 'y')}
          stroke="#00ff88"
          strokeWidth="2"
          strokeLinecap="round"
          filter="url(#textGlow)"
        />

        {/* Current State Marker */}
        <circle
          cx={toSVG(last.valence, 'x')}
          cy={toSVG(last.arousal, 'y')}
          r={5}
          fill="#bf00ff"
          className="animate-pulse"
        />
        <circle
          cx={toSVG(last.valence, 'x')}
          cy={toSVG(last.arousal, 'y')}
          r={10}
          fill="none"
          stroke="#bf00ff"
          strokeWidth="1"
          strokeOpacity="0.4"
          className="animate-ping"
        />

        {/* HUD Data Labels */}
        <g className="font-mono text-[9px] fill-purple-300" style={{ filter: 'url(#textGlow)' }}>
          <text x={padding} y={padding - 10}>VEL_MAG: {last.velocity.toFixed(3)}</text>
          <text x={size - padding - 65} y={padding - 10}>HDG: {bearing}°</text>
          <text x={padding} y={size - padding + 20}>VAL: {last.valence.toFixed(2)}</text>
          <text x={size - padding - 45} y={size - padding + 20}>ARS: {last.arousal.toFixed(2)}</text>
        </g>
      </svg>

      {/* Bottom Telemetry Bar */}
      <div className="mt-2 flex justify-between border-t border-purple-900/50 pt-2 px-1">
        <div className="flex flex-col gap-0.5">
          <span className="text-[6px] text-purple-600 font-mono uppercase tracking-widest">Dynamics_Buffer</span>
          <div className="flex gap-0.5">
            {[1, 1, 0, 1, 0, 1, 1, 1, 0].map((v, i) => (
              <div key={i} className={`h-1 w-1 ${v ? 'bg-neon-purple' : 'bg-purple-900'}`} />
            ))}
          </div>
        </div>
        <div className="text-[8px] font-mono text-neon-green/70 flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-neon-green animate-pulse shadow-neon-sm" />
          SIGNAL_LOCKED
        </div>
      </div>
    </div>
  );
};

export default EmotionVectorField;
