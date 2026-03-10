const EmotionVolatility = ({ data }) => {

  let changes = 0

  for (let i = 1; i < data.length; i++) {
    if (data[i].emotion !== data[i - 1].emotion)
      changes++
  }

  const volatility = (data.length > 0 ? (changes / data.length * 100) : 0).toFixed(1)

  return (

    <div className="flex flex-col items-center justify-center h-32">

      <div className="text-xs text-purple-400">
        EMOTION VOLATILITY
      </div>

      <div className="text-3xl text-neon-purple font-bold">
        {volatility}%
      </div>

    </div>

  )
}

export default EmotionVolatility
