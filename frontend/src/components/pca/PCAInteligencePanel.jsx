import PCAScatter3D from "./PCAScatter3D"
import PCAVarianceSpectrum from "./PCAVarianceSpectrum"
import PCACumulativeVariance from "./PCACumulativeVariance"
import PCALegend from "./PCALegend"


export default function PCAInteligencePanel({ data }) {

  if (!data) return null

  const variance = data.variance || []

  const explained =
    variance.slice(0,3).reduce((a,b)=>a+b,0)

  const sessions = data.points?.length || 0

  return (

    <div style={{ padding: 20 }}>

      <h3>PCA Intelligence</h3>

      <p>
        Sessions analyzed:
        <strong> {sessions}</strong>
      </p>

      <p>
        Variance explained by first 3 components:
        <strong> {(explained*100).toFixed(2)}%</strong>
      </p>

      <p>
        PCA compresses emotional embeddings
        into a 3D cognitive manifold.
      </p>

    </div>
  )
}

