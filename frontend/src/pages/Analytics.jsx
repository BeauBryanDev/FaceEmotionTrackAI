// =============================================================================
// frontend/src/pages/Analytics.jsx
//
// Biometric analytics dashboard. Consumes:
//   GET    /api/v1/analytics/pca             -> 3D PCA scatter plot payload
//   GET    /api/v1/analytics/session/history -> session embedding list
//   DELETE /api/v1/analytics/session         -> clear session embeddings
//
// Layout:
//   Row 1: PCA stats cards (total points, variance explained, user point)
//   Row 2: PCA scatter plot (SVG, interactive hover)
//   Row 3: Session history table + clear button
// =============================================================================

import React, { useEffect, useState, useMemo, useRef, useCallback } from 'react';
import { getPcaVisualization } from '../api/analytics';
import { BarChart2, Info, Target, Crosshair } from 'lucide-react';
import Text from '../components/ui/Text';

// -----------------------------------------------------------------------------
// TACTICAL UI COMPONENTS
// Military-grade cyberpunk containers with neon accents.
// -----------------------------------------------------------------------------
const TacticalCard = ({ children, className = "" }) => (
  <div className={`relative bg-surface-1 border border-purple-800/50 p-5 shadow-[inset_0_0_20px_rgba(74,0,128,0.15)] ${className}`}>
    {/* HUD Corner Brackets */}
    <div className="absolute top-0 left-0 w-3 h-3 border-t border-l border-purple-400" />
    <div className="absolute bottom-0 right-0 w-3 h-3 border-b border-r border-purple-400" />
    {children}
  </div>
);

const StatCard = ({ label, value, sub, highlightColor = 'text-neon-purple' }) => (
  <TacticalCard>
    <div className="font-mono text-[10px] tracking-[0.2em] text-purple-400 mb-2">
      {label}
    </div>
    <div className={`font-display text-4xl font-black leading-none drop-shadow-[0_0_12px_currentColor] ${highlightColor}`}>
      {value}
    </div>
    {sub && (
      <div className="font-mono text-[10px] text-purple-500 mt-2 tracking-wider">
        {sub}
      </div>
    )}
  </TacticalCard>
);

// -----------------------------------------------------------------------------
// PCA SCATTER PLOT ENGINE (SVG)
// Projects high-dimensional face embeddings into 2D latent space.
// -----------------------------------------------------------------------------
const PCAScatter = ({ points }) => {
  const svgRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);

  // SVG Coordinate System Dimensions
  const W = 800;
  const H = 400;
  const PAD = 40;

  // Dynamic boundary calculation to auto-scale the scatter plot
  const bounds = useMemo(() => {
    if (!points.length) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of points) {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    }
    // Add 15% padding to the mathematical bounds for visual breathing room
    const dx = (maxX - minX) * 0.15 || 0.15;
    const dy = (maxY - minY) * 0.15 || 0.15;
    return { minX: minX - dx, maxX: maxX + dx, minY: minY - dy, maxY: maxY + dy };
  }, [points]);

  // Project mathematical coordinates into SVG space
  const project = (p) => ({
    cx: PAD + ((p.x - bounds.minX) / (bounds.maxX - bounds.minX)) * (W - PAD * 2),
    cy: PAD + ((p.y - bounds.minY) / (bounds.maxY - bounds.minY)) * (H - PAD * 2),
  });

  const getPointTheme = (p) => {
    if (p.is_current_user) return { color: '#00ff88', radius: 6, glow: 'glow-green' };
    if (p.source === 'registered') return { color: '#bf00ff', radius: 4, glow: 'glow-purple' };
    return { color: '#cc44ff', radius: 3, glow: 'glow-faded' };
  };

  return (
    <div className="relative w-full overflow-hidden bg-surface-2 border border-purple-900">
      
      {/* Target Crosshair Decoration */}
      <Crosshair className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 text-purple-900/30 pointer-events-none animate-[spin_60s_linear_infinite]" />

      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-auto cursor-crosshair"
        onMouseLeave={() => setTooltip(null)}
      >
        {/* Tactical Grid Overlay */}
        {[0.25, 0.5, 0.75].map((t) => (
          <g key={`grid-${t}`}>
            <line
              x1={PAD + t * (W - PAD * 2)} y1={PAD}
              x2={PAD + t * (W - PAD * 2)} y2={H - PAD}
              stroke="rgba(170,0,255,0.15)" strokeWidth="1" strokeDasharray="4 4"
            />
            <line
              x1={PAD} y1={PAD + t * (H - PAD * 2)}
              x2={W - PAD} y2={PAD + t * (H - PAD * 2)}
              stroke="rgba(170,0,255,0.15)" strokeWidth="1" strokeDasharray="4 4"
            />
          </g>
        ))}

        {/* Axes Labels */}
        <text x={W / 2} y={H - 10} textAnchor="middle" fill="rgba(170,0,255,0.5)" className="font-mono text-xs tracking-widest">
          PRINCIPAL COMPONENT 1 (PC1)
        </text>
        <text x={15} y={H / 2} textAnchor="middle" fill="rgba(170,0,255,0.5)" className="font-mono text-xs tracking-widest" transform={`rotate(-90, 15, ${H / 2})`}>
          PRINCIPAL COMPONENT 2 (PC2)
        </text>

        {/* Data Points Rendering */}
        {points.map((p, i) => {
          const { cx, cy } = project(p);
          const theme = getPointTheme(p);
          
          return (
            <g key={i}
              onMouseEnter={(e) => {
                const rect = svgRef.current.getBoundingClientRect();
                // Calculate relative position to SVG container for tooltip
                const scaleX = W / rect.width;
                const scaleY = H / rect.height;
                setTooltip({
                  x: (e.clientX - rect.left) * scaleX,
                  y: (e.clientY - rect.top) * scaleY,
                  point: p,
                });
              }}
              className="transition-transform duration-200 hover:scale-150 origin-center cursor-crosshair"
              style={{ transformOrigin: `${cx}px ${cy}px` }}
            >
              {/* Radar pulse for current user */}
              {p.is_current_user && (
                <circle cx={cx} cy={cy} r={theme.radius * 2.5} fill="none" stroke={theme.color} strokeWidth="1" className="animate-ping opacity-75" />
              )}
              {/* Core point */}
              <circle cx={cx} cy={cy} r={theme.radius} fill={theme.color} className={theme.glow} />
            </g>
          );
        })}

        {!points.length && (
          <text x={W / 2} y={H / 2} textAnchor="middle" fill="rgba(170,0,255,0.5)" className="font-mono text-sm tracking-widest">
            AWAITING VECTOR INGESTION...
          </text>
        )}
      </svg>

      {/* Floating Tactical Tooltip */}
      {tooltip && (
        <div 
          className="absolute z-50 bg-purple-950/95 border border-neon-purple p-3 shadow-neon-md pointer-events-none backdrop-blur-sm"
          style={{
            // Convert SVG coordinates back to CSS percentage for absolute positioning
            left: `${(tooltip.x / W) * 100}%`,
            top: `${(tooltip.y / H) * 100}%`,
            transform: 'translate(-50%, -120%)'
          }}
        >
          <div className="absolute bottom-[-6px] left-1/2 -translate-x-1/2 w-3 h-3 bg-purple-950 border-b border-r border-neon-purple rotate-45" />
          
          <div className={`font-mono text-xs font-bold tracking-widest mb-1 ${tooltip.point.is_current_user ? 'text-green-400' : 'text-neon-purple'}`}>
            {tooltip.point.is_current_user ? 'TARGET: OPERATOR' : `TARGET: ENTITY ${tooltip.point.user_id}`}
          </div>
          <div className="font-mono text-[10px] text-purple-300 tracking-wider flex flex-col gap-0.5">
            <span>SRC: {tooltip.point.source.toUpperCase()}</span>
            <span>PC1: {tooltip.point.x.toFixed(4)}</span>
            <span>PC2: {tooltip.point.y.toFixed(4)}</span>
            <span>PC3: {tooltip.point.z.toFixed(4)}</span>
          </div>
        </div>
      )}

      {/* Military Legend */}
      <div className="absolute bottom-4 right-4 flex flex-col gap-2 bg-surface-1/80 p-3 border border-purple-800/50 backdrop-blur-md">
        {[
          { color: 'bg-green-400', label: 'OPERATOR ID' },
          { color: 'bg-neon-purple', label: 'DB ENTITIES' },
          { color: 'bg-purple-500', label: 'TRANSIENT SESSIONS' },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-2">
            <div className={`w-2 h-2 ${color} rounded-full shadow-[0_0_5px_currentColor]`} />
            <span className="font-mono text-[9px] tracking-widest text-purple-200">
              {label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

// -----------------------------------------------------------------------------
// MAIN ANALYTICS VIEW
// Unifies V1 Header/Footer with V2 SVG rendering capabilities.
// -----------------------------------------------------------------------------
const Analytics = () => {
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pcaHint, setPcaHint] = useState(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    setPcaHint(null);
    try {
      // Fetch only PCA visualization data, ignoring deprecated cache logs
      const pcaData = await getPcaVisualization();
      setPayload(pcaData);
    } catch (e) {
      const detail = e?.response?.data?.detail || '';
      if (e?.response?.status === 422 && detail.toLowerCase().includes('insufficient')) {
        setPcaHint(detail);
      } else {
        setError('FATAL EXCEPTION: PCA MANIFOLD GENERATION FAILED.');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const points = payload?.points || [];
  const totalVariance = payload?.total_variance || 0;
  const variancePerComp = payload?.explained_variance || [];
  const currentUserPts = points.filter((p) => p.is_current_user).length;

  return (
    <div className="p-4 sm:p-6 min-h-[calc(100vh-80px)] bg-surface-0 bg-cyber-grid flex flex-col gap-6">
      
      {/* GLOBAL GLOW ANIMATIONS */}
      <style>{`
        .glow-green { filter: drop-shadow(0 0 8px rgba(0, 255, 136, 0.9)); }
        .glow-purple { filter: drop-shadow(0 0 6px rgba(191, 0, 255, 0.8)); }
        .glow-faded { filter: drop-shadow(0 0 4px rgba(204, 68, 255, 0.4)); }
      `}</style>

      {/* V1 HEADER RETAINED AND UPGRADED */}
      <div className="border-b border-purple-800 pb-4 flex flex-col md:flex-row md:justify-between md:items-end gap-3">
        <div>
          <Text variant="h2" glow className="flex items-center gap-3 text-purple-100">
            <Target className="w-7 h-7 text-neon-purple" />
            PCA ANALYTICS
          </Text>
          <Text variant="mono" className="mt-2 text-purple-400">
            BIOMETRIC LATENT SPACE PROJECTION // DIMENSIONALITY REDUCTION
          </Text>
        </div>
        <div className="text-right hidden md:block">
          <Text variant="mono" className="text-neon-purple font-bold">
            VECTORS: {payload?.total_points || 0}
          </Text>
          <Text variant="subtext" className="text-purple-500">
            VARIANCE: {(totalVariance * 100).toFixed(1)}%
          </Text>
        </div>
      </div>

      {loading && (
        <div className="flex-1 flex flex-col items-center justify-center min-h-[260px] sm:min-h-[400px]">
          <BarChart2 className="w-12 h-12 text-neon-purple animate-pulse mb-4" />
          <Text variant="mono" className="text-purple-400 animate-pulse tracking-[0.3em]">
            COMPUTING EIGENVECTORS...
          </Text>
        </div>
      )}

      {error && !loading && (
        <TacticalCard className="border-red-900 bg-red-950/20 text-center py-10">
          <Text variant="mono" className="text-red-500 tracking-widest">{error}</Text>
        </TacticalCard>
      )}

      {pcaHint && !loading && (
        <TacticalCard className="border-yellow-900/50 bg-yellow-900/10">
          <div className="flex flex-col sm:flex-row gap-4 items-start">
            <Info className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
            <div>
              <Text variant="mono" className="text-yellow-400 mb-1">{pcaHint}</Text>
              <Text variant="subtext" className="text-purple-400">
                SYSTEM REQUIREMENT: PCA MATRIX REQUIRES A MINIMUM OF 3 VECTORS TO ESTABLISH LATENT SPACE.
                ENROLL ADDITIONAL SUBJECTS OR CAPTURE LIVE TELEMETRY.
              </Text>
            </div>
          </div>
        </TacticalCard>
      )}

      {!loading && !error && !pcaHint && payload && (
        <div className="flex flex-col gap-6">
          
          {/* METRICS ROW */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <StatCard 
              label="TOTAL EMBEDDINGS" 
              value={payload.total_points} 
              sub="POINTS IN LATENT MANIFOLD"
              highlightColor="text-purple-200"
            />
            <StatCard 
              label="EXPLAINED VARIANCE" 
              value={`${(totalVariance * 100).toFixed(1)}%`} 
              sub="INFORMATION RETAINED BY 3D PROJECTION"
              highlightColor="text-green-400"
            />
            <StatCard 
              label="OPERATOR VECTORS" 
              value={currentUserPts} 
              sub="AUTHORIZED SESSIONS DETECTED"
              highlightColor="text-neon-purple"
            />
          </div>

          {/* MAIN PROJECTION HUD */}
          <TacticalCard>
            <div className="flex items-center justify-between mb-4 border-b border-purple-900/50 pb-2">
              <Text variant="mono" className="text-purple-300 flex items-center gap-2">
                <Crosshair className="w-4 h-4" /> SPATIAL PROJECTION
              </Text>
              <div className="hidden md:flex gap-4">
                {variancePerComp.slice(0, 3).map((v, i) => (
                  <Text key={`var-${i}`} variant="subtext" className="text-purple-500">
                    PC{i + 1}: {(v * 100).toFixed(1)}%
                  </Text>
                ))}
              </div>
            </div>
            
            <PCAScatter points={points} />
          </TacticalCard>
        </div>
      )}

      {/* V1 FOOTER RETAINED */}
      <div className="mt-auto pt-6 border-t border-purple-800/50 text-center">
        <Text variant="subtext" className="text-purple-600">
          [ SYS.RES.PCA_COMPUTATION_COMPLETE ] // LATENT SPACE MAPPED AND SECURED
        </Text>
      </div>

    </div>
  );
};

export default Analytics;
