import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"

const EmotionActivityTimeline = ({ data }) => {
  const emotions = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise"
  ]

  const formatted = data.map((d, i) => ({
    index: i,
    emotion: d.emotion // Fixed: timeline endpoint returns 'emotion' directly
  }))

  return (
    <div className="h-40">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 10, right: 10, bottom: 0, left: 60 }}>
          <XAxis
            dataKey="index"
            hide
          />
          <YAxis
            dataKey="emotion"
            type="category"
            categories={emotions}
            tick={{ fill: 'rgba(170,0,255,0.7)', fontSize: 9, fontFamily: 'monospace' }}
            axisLine={{ stroke: 'rgba(170,0,255,0.2)' }}
            tickLine={false}
          />
          <Tooltip
            cursor={{ strokeDasharray: '3 3', stroke: 'rgba(170,0,255,0.5)' }}
            contentStyle={{ backgroundColor: '#1a0033', border: '1px solid #cc44ff', fontSize: '10px', fontFamily: 'monospace' }}
          />
          <Scatter
            data={formatted}
            fill="#cc44ff"
            line={{ stroke: 'rgba(170,0,255,0.3)', strokeWidth: 1 }}
            shape="diamond"
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

export default EmotionActivityTimeline
