import { useEffect, useState } from "react"
import { getPcaVisualization } from "../api/analytics"

export default function usePCAData() {

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const loadPCA = async () => {
    try {
      setLoading(true)

      const response = await getPcaVisualization({
        include_sessions: true,
        session_limit: 200
      })

      setData(response)

    } catch (err) {
      setError(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadPCA()
  }, [])

  return {
    data,
    loading,
    error,
    reload: loadPCA
  }
}
