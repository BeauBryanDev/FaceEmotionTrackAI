export default function PCALegend({ data }) {

  const clusters = data?.clusters || []

  return (

    <div style={{ padding: 20 }}>

      <h3>Emotion Clusters</h3>

      {clusters.map(c => (

        <div
          key={c.id}
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 8
          }}
        >

          <span>{c.label}</span>

          <span>{c.size}</span>

        </div>

      ))}

    </div>
  )
}
