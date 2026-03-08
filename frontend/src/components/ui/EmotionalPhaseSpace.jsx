import { LineChart, Line, XAxis, YAxis, Tooltip } from "recharts"

export default function EmotionalPhaseSpace({ data }) {
  return (
    <div className="bg-black/40 border border-purple-700 p-4 rounded-xl">
      <h2 className="text-purple-400 mb-3 font-mono">
        Emotional Phase Space
      </h2>

      <LineChart width={420} height={320} data={data}>
        <XAxis dataKey="valence" type="number" domain={[-1,1]} />
        <YAxis dataKey="arousal" type="number" domain={[-1,1]} />
        <Tooltip />

        <Line
          type="monotone"
          dataKey="arousal"
          stroke="#a855f7"
          dot={{ r:4 }}
        />
      </LineChart>
    </div>
  )
}
