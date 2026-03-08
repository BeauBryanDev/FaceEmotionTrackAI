export default function EmotionFlowField() {

  const vectors = []

  for (let x=-1; x<=1; x+=0.4){
    for (let y=-1; y<=1; y+=0.4){

      vectors.push({
        x,
        y,
        vx: -y,
        vy: x
      })
    }
  }

  return (
    <svg width="320" height="320">

      {vectors.map((v,i)=>{

        const startX = 160 + v.x * 120
        const startY = 160 - v.y * 120

        const endX = startX + v.vx * 20
        const endY = startY - v.vy * 20

        return (
          <line
            key={i}
            x1={startX}
            y1={startY}
            x2={endX}
            y2={endY}
            stroke="#a855f7"
            strokeWidth="2"
          />
        )
      })}

    </svg>
  )
}
