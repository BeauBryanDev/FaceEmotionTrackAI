const EmotionDrift = ({ dynamics }) => {

  if(!dynamics || dynamics.length === 0) return null

  const avgDrift =
    dynamics.reduce((a,b)=>a+b.drift,0)/dynamics.length

  return (

  <div className="flex flex-col items-center justify-center h-36">

    <div className="text-xs text-purple-400">
      EMOTIONAL DRIFT
    </div>

    <div className="text-3xl text-purple-300 font-bold">
      {avgDrift.toFixed(3)}
    </div>

  </div>

  )
}

export default EmotionDrift
