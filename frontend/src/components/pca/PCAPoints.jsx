import { useMemo } from "react"

export default function PCAPoints({ points }) {

  const geometry = useMemo(() => {

    const vertices = []

    points.forEach(p => {
      vertices.push(p.pc1, p.pc2, p.pc3)
    })

    return new Float32Array(vertices)

  }, [points])

  return (

    <points>

      <bufferGeometry>

        <bufferAttribute
          attach="attributes-position"
          count={geometry.length / 3}
          array={geometry}
          itemSize={3}
        />

      </bufferGeometry>

      <pointsMaterial
        color="#a855f7"
        size={0.05}
        sizeAttenuation
      />

    </points>

  )
}
