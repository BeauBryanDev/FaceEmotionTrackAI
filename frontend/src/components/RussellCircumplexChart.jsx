import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from "recharts"
import { getQuadrant } from "../utils/russellMapping"

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const { valence, arousal } = payload[0].payload
    const quad = getQuadrant(valence, arousal)

    return (
      <div className="bg-purple-950/95 border border-neon-purple p-3 backdrop-blur-md shadow-[0_0_15px_rgba(170,0,255,0.4)] font-mono">
        <div className="flex items-center gap-2 mb-2 border-b border-purple-800 pb-1">
          <div className="w-2 h-2 bg-neon-purple animate-pulse" />
          <span className="text-purple-200 text-[10px] font-bold tracking-widest uppercase">
            AFFECT_COORDINATES
          </span>
        </div>

        <div className="space-y-1">
          <div className="flex justify-between gap-4 text-[10px]">
            <span className="text-purple-500">VALENCE (X):</span>
            <span className="text-neon-purple">{valence.toFixed(3)}</span>
          </div>
          <div className="flex justify-between gap-4 text-[10px]">
            <span className="text-purple-500">AROUSAL (Y):</span>
            <span className="text-neon-purple">{arousal.toFixed(3)}</span>
          </div>
          <div className="mt-2 pt-1 border-t border-purple-900/50">
            <div className="text-[9px] text-purple-400 mb-0.5">TACTICAL_STATE:</div>
            <div
              className="text-[11px] font-black uppercase"
              style={{ color: quad.color }}
            >
              {quad.label}
            </div>
          </div>
        </div>
      </div>
    )
  }
  return null
}

const RussellCircumplexChart = ({ data }) => {
  //Main Graph for Russell Circumplex Model of Affect
  const denseTicks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
  return (

    <div className="h-80 w-full relative">
      {/* Grid Labels (Static) */}
      <div className="absolute top-0 left-0 right-0 bottom-0 pointer-events-none opacity-20 font-mono text-[8px] text-purple-600 p-2 flex flex-col justify-between">
        <div className="flex justify-between">
          <span>[ High Arousal ]</span>
          <span>[ High Arousal ]</span>
        </div>
        <div className="flex justify-between mt-auto">
          <span>[ Low Arousal ]</span>
          <span>[ Low Arousal ]</span>
        </div>
      </div>

      <div className="absolute top-0 left-0 right-0 bottom-0 pointer-events-none opacity-10 font-mono text-[8px] text-purple-600 p-2 flex items-center justify-between">
        <span>[ Negative Valence ]</span>
        <span>[ Positive Valence ]</span>
      </div>

      <ResponsiveContainer>
        <ScatterChart>
          <CartesianGrid stroke="#4b1d6a" strokeOpacity={0.55} strokeDasharray="2 2" />

          <XAxis
            type="number"
            dataKey="valence"
            domain={[-1, 1]}
            ticks={denseTicks}
            name="Valence"
            axisLine={{ stroke: "#2e0f47", strokeWidth: 2 }}
            tickLine={false}
            tick={{ fill: "#b88cff", fontSize: 10 }}
          />

          <YAxis
            type="number"
            dataKey="arousal"
            domain={[-1, 1]}
            ticks={denseTicks}
            name="Arousal"
            axisLine={{ stroke: "#2e0f47", strokeWidth: 2 }}
            tickLine={false}
            tick={{ fill: "#b88cff", fontSize: 10 }}
          />

          <ReferenceLine x={0} stroke="#2e0f47" strokeWidth={2} />
          <ReferenceLine y={0} stroke="#2e0f47" strokeWidth={2} />

          <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#bf00ff', strokeWidth: 1, strokeDasharray: '3 3' }} />

          <Scatter
            data={data}
            fill="#cc44ff"
            line={{ stroke: 'rgba(191,0,255,0.4)', strokeWidth: 1 }}
            shape="circle"
          />

        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

export default RussellCircumplexChart
