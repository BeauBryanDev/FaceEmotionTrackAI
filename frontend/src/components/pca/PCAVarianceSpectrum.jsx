import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from "recharts"

export default function PCAVarianceSpectrum({ data }) {

  const variance = data?.variance || []

  const chartData = variance.map((v,i)=>({
    component: `PC${i+1}`,
    variance: v
  }))

  return (

    <ResponsiveContainer width="100%" height={220}>

      <BarChart data={chartData}>

        <XAxis dataKey="component" />
        <YAxis />

        <Tooltip />

        <Bar
          dataKey="variance"
          fill="#9333ea"
        />

      </BarChart>

    </ResponsiveContainer>

  )
}
