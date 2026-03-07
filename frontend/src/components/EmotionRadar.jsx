import React, { useMemo } from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip
} from 'recharts';

/**
 * CUSTOM HUD TOOLTIP
 * Styled as a floating terminal fragment
 */
const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-purple-950/95 border-l-2 border-neon-purple p-3 backdrop-blur-md shadow-[0_0_20px_rgba(170,0,255,0.3)] animate-in fade-in zoom-in duration-200">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-1.5 h-1.5 bg-neon-purple animate-pulse" />
          <p className="text-purple-200 font-display text-[10px] font-bold tracking-widest">
            {payload[0].payload.subject} // NEURAL_INDEX
          </p>
        </div>
        <p className="text-neon-purple font-mono text-xs font-black shadow-neon-sm">
          INTENSITY: {(payload[0].value * 100).toFixed(1)}%
        </p>
      </div>
    );
  }
  return null;
};

/**
 * NEON VERTEX DOT
 * Custom dot component for vertex glow
 */
const RenderCustomDot = (props) => {
  const { cx, cy, payload, value } = props;
  if (value < 0.05) return null; // Don't show if intensity is negligible

  return (
    <g>
      <circle cx={cx} cy={cy} r={4} fill="#bf00ff" filter="url(#vertexGlow)" />
      <circle cx={cx} cy={cy} r={2} fill="#f0ccff" />
    </g>
  );
};

const EmotionRadar = ({ emotionScores }) => {
  // Transform scores into HUD-compatible data
  const data = useMemo(() => {
    const defaultData = [
      { subject: 'ANG', A: 0 }, { subject: 'CON', A: 0 },
      { subject: 'DIS', A: 0 }, { subject: 'FEA', A: 0 },
      { subject: 'HAP', A: 0 }, { subject: 'NEU', A: 0 },
      { subject: 'SAD', A: 0 }, { subject: 'SUR', A: 0 },
    ];

    if (!emotionScores || Object.keys(emotionScores).length === 0) return defaultData;

    const labels = {
      "Anger": "ANG", "Contempt": "CON", "Disgust": "DIS", "Fear": "FEA",
      "Happiness": "HAP", "Neutral": "NEU", "Sadness": "SAD", "Surprise": "SUR"
    };

    return Object.keys(emotionScores).map(key => ({
      subject: labels[key] || key.substring(0, 3).toUpperCase(),
      A: emotionScores[key],
    }));
  }, [emotionScores]);

  return (
    <div className="w-full h-full min-h-[280px] relative flex flex-col items-center justify-center bg-surface-1/40 border border-purple-900/50 overflow-hidden group">

      {/* HUD CORNER ACCENTS */}
      <div className="absolute top-0 left-0 w-8 h-8 border-t border-l border-purple-500/30 transition-all group-hover:border-neon-purple/60" />
      <div className="absolute top-0 right-0 w-8 h-8 border-t border-r border-purple-500/30 transition-all group-hover:border-neon-purple/60" />
      <div className="absolute bottom-0 left-0 w-8 h-8 border-b border-l border-purple-500/30 transition-all group-hover:border-neon-purple/60" />
      <div className="absolute bottom-0 right-0 w-8 h-8 border-b border-r border-purple-500/30 transition-all group-hover:border-neon-purple/60" />

      {/* BACKGROUND SCANNING GRID LINES */}
      <div className="absolute inset-0 opacity-10 pointer-events-none overflow-hidden">
        <div className="absolute inset-0 bg-cyber-grid bg-[length:20px_20px]" />
        <div className="absolute w-full h-[1px] bg-neon-purple/40 top-0 animate-[scan_4s_linear_infinite]" />
      </div>

      {/* MODULE HEADER */}
      <div className="absolute top-3 left-4 flex flex-col gap-0.5 z-10">
        <div className="flex items-center gap-2">
          <div className="w-2 h-0.5 bg-neon-purple" />
          <span className="text-purple-400 font-mono text-[9px] tracking-[0.3em] font-bold">
            NEURALCORE.RADAR_V4
          </span>
        </div>
        <span className="text-purple-600 font-mono text-[7px] pl-4">STATUS: ANALYZING_STREAM</span>
      </div>

      <ResponsiveContainer width="100%" height="90%">
        <RadarChart cx="50%" cy="50%" outerRadius="65%" data={data}  className="animate-[neuralPulse_3s_ease-in-out_infinite]">

          <defs>
            {/* Primary Energy Fill */}
            <radialGradient id="energyFill" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
              <stop offset="0%" stopColor="#bf00ff" stopOpacity={0.6} />
              <stop offset="90%" stopColor="#7a00cc" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#4b0082" stopOpacity={0.05} />
            </radialGradient>

            {/* Polygon Glow Filter */}
            <filter id="neonBlur" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="3.5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>

            {/* Vertex Glow */}
            <filter id="vertexGlow">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feFlood floodColor="#bf00ff" result="color" />
              <feComposite in="color" in2="blur" operator="in" />
              <feMerge>
                <feMergeNode />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* POLAR GRID - TARGETING RETICLE STYLE */}
          <PolarGrid
            stroke="#6b21a8"
            strokeOpacity={0.3}
            strokeDasharray="4 2"
            polarAngles={[0, 45, 90, 135, 180, 225, 270, 315]}
          />
          <PolarRadiusAxis
            domain={[0, 1]}
            tick={{ fill: '#7c3aed', fontSize: 8, fontFamily: 'Share Tech Mono' }}
            axisLine={{ stroke: '#6b21a8', strokeOpacity: 0.2 }}
            tickCount={4}
          />

          {/* LABELS */}
          <PolarAngleAxis
            dataKey="subject"
            tick={{
              fill: '#d8b4fe',
              fontSize: 10,
              fontFamily: 'Share Tech Mono',
              fontWeight: 900,
              letterSpacing: '0.1em'
            }}
          />

          <PolarRadiusAxis
            domain={[0, 1]}
            tick={false}
            axisLine={{ stroke: '#6b21a8', strokeOpacity: 0.2 }}
          />

          <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#bf00ff', strokeWidth: 1 }} />

          {/* GHOST / AMBIENT LAYER & INNER SHADOW */}
          <Radar
            dataKey="A"
            stroke="none"
            fill="#7a00cc"
            fillOpacity={0.1}
            isAnimationActive={false}
          />

          {/* ACTION LAYER - NEON EDGE */}
          <Radar
            name="Emotions"
            dataKey="A"
            stroke="#bf00ff"
            strokeWidth={2}
            fill="url(#energyFill)"
            fillOpacity={0.8}
            dot={<RenderCustomDot />}
            filter="url(#neonBlur)"
            isAnimationActive={true}
            animationDuration={400}
            animationEasing="ease-out"
          />

          <Radar
          dataKey="A"
          stroke="#bf00ff"
          strokeWidth={4}
          fill="none"
          strokeOpacity={0.15}
          isAnimationActive={false}
        />

          <circle
            cx="50%"
            cy="50%"
            r="4"
            fill="#bf00ff"
            opacity="0.8"
          />

          <circle
            cx="50%"
            cy="50%"
            r="10"
            stroke="#bf00ff"
            strokeOpacity="0.3"
            fill="none"
          />

        </RadarChart>
      </ResponsiveContainer>

      {/* FOOTER DATA STREAM (Decorative) */}
      <div className="absolute bottom-2 right-3 flex items-center gap-4 opacity-40 pointer-events-none">
        <div className="flex flex-col items-end">
          <div className="h-1 w-12 bg-purple-900 rounded-full overflow-hidden">
            <div className="h-full bg-neon-purple w-2/3 animate-[loading_2s_infinite]" />
          </div>
          <span className="text-[6px] text-purple-400 font-mono mt-0.5 uppercase">Syncing...</span>
        </div>
        <div className="font-mono text-[8px] text-purple-500 border-l border-purple-800/50 pl-2">
          MTX_PROT: V3.2
        </div>
      </div>

      <style>{`
        @keyframes scan {
          0% { transform: translateY(-100%); opacity: 0; }
          10% { opacity: 0.4; }
          90% { opacity: 0.4; }
          100% { transform: translateY(300px); opacity: 0; }
        }
        @keyframes loading {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
};

export default EmotionRadar;
