import usePCAData from "../hooks/usePCAData"

import PCAScatter3D from "../components/pca/PCAScatter3D"
import PCAVarianceSpectrum from "../components/pca/PCAVarianceSpectrum"
import PCAInteligencePanel from "../components/pca/PCAInteligencePanel"



export default function PCAAnalytics() {

  const { data, loading, error } = usePCAData()

  if (loading) return <div className="pca-loading">Loading PCA analytics...</div>

  if (error) return <div className="pca-error">Failed to load PCA data</div>

  return (

    <div className="pca-dashboard p-4 sm:p-6">

      <header className="pca-header text-center sm:text-left">
        <h1>PCA Emotional Embedding Space</h1>
        <p>Dimensionality reduction of emotion embeddings</p>
      </header>

      <div className="pca-grid">

        <div className="pca-card pca-3d">
          <PCAScatter3D data={data} />
        </div>

        <div className="pca-card">
          <PCAVarianceSpectrum data={data} />
        </div>

        <div className="pca-card">
          <PCAInteligencePanel data={data} />
        </div>

      </div>

    </div>
  )
}
