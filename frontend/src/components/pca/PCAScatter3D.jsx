import { Canvas } from "@react-three/fiber"
import { OrbitControls, Grid } from "@react-three/drei"
import PCAPoints from "./PCAPoints"

export default function PCAScatter3D({ data }) {

  if (!data) return null

  return (
    <div style={{ height: "100%", width: "100%" }}>
      <Canvas camera={{ position: [4,4,4] }}>

        <ambientLight intensity={0.5} />
        <pointLight position={[10,10,10]} />

        <Grid args={[10,10]} />

        <PCAPoints points={data.points} />

        <OrbitControls />

      </Canvas>
    </div>
  )
}
