import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from "recharts"

export default function PCACumulativeVariance({ data }) {

  const variance = data?.variance || []

  let cumulative = 0

  const chartData = variance.map((v,i)=>{
    cumulative += v

    return {
      component: `PC${i+1}`,
      cumulative
    }
  })

  return (

    <ResponsiveContainer width="100%" height={220}>

      <LineChart data={chartData}>

        <XAxis dataKey="component" />
        <YAxis />

        <Tooltip />

        <Line
          type="monotone"
          dataKey="cumulative"
          stroke="#c084fc"
        />

      </LineChart>

    </ResponsiveContainer>

  )
}
