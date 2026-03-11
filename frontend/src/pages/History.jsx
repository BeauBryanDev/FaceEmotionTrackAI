import React, { useState, useEffect, useCallback } from 'react';
import { getEmotionHistory } from '../api/emotions';
import { History as HistoryIcon, Filter, ChevronLeft, ChevronRight, AlertTriangle, Database } from 'lucide-react';

const VALID_EMOTIONS = [
  "ALL", "Anger", "Contempt", "Disgust", "Fear",
  "Happiness", "Neutral", "Sadness", "Surprise"
];

const History = () => {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Estados de paginación y filtros
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalRecords, setTotalRecords] = useState(0);
  const [emotionFilter, setEmotionFilter] = useState("ALL");

  // Función para consumir el endpoint REST
  const fetchHistory = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Construir los parámetros de la query
      const params = {
        page: page,
        page_size: 10,
      };
      
      if (emotionFilter !== "ALL") {
        params.emotion_filter = emotionFilter;
      }

      // Llamada Axios al backend (Ajusta la URL base si la tienes configurada en tu axios.js)
      const data = await getEmotionHistory(params);

      setRecords(data.records);
      setTotalPages(data.total_pages);
      setTotalRecords(data.total_records);
      
    } catch (err) {
      console.error("Error fetching history:", err);
      setError("Fallo al conectar con la base de datos neuronal.");
    } finally {
      setLoading(false);
    }
  }, [page, emotionFilter]);

  // Disparar la petición cuando cambia la página o el filtro
  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  // Formatear el timestamp ISO 8601 a formato Terminal Cyberpunk
  const formatCyberTime = (isoString) => {
    const date = new Date(isoString);
    return date.toISOString().replace('T', ' ').substring(0, 19);
  };

  // Manejador del cambio de filtro (resetea a la página 1)
  const handleFilterChange = (e) => {
    setEmotionFilter(e.target.value);
    setPage(1); 
  };

  return (
    <div className="p-4 sm:p-6 min-h-[calc(100vh-80px)] bg-surface-0 bg-cyber-grid font-body flex flex-col gap-6">
      
      {/* CABECERA DEL MÓDULO */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-purple-800 pb-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-display font-black text-purple-200 tracking-widest flex items-center gap-3">
            <Database className="w-6 h-6 text-neon-purple" />
            TELEMETRY ARCHIVE
          </h1>
          <p className="font-mono text-xs text-purple-400 mt-1">
            GETTING HISTORICAL EMOTIONAL DATA // {totalRecords} ENTRIES LOCATED
          </p>
        </div>

        {/* CONTROLES DE FILTRADO */}
        <div className="flex flex-col sm:flex-row sm:items-center gap-2 bg-surface-1 border border-purple-700 p-2 shadow-neon-sm w-full sm:w-auto">
          <Filter className="w-4 h-4 text-purple-400" />
          <span className="font-mono text-xs text-purple-300 mr-2">STATE_FILTER:</span>
          <select 
            value={emotionFilter}
            onChange={handleFilterChange}
            className="bg-purple-950 border border-purple-600 text-neon-purple font-mono text-sm px-2 py-1 outline-none focus:border-neon-purple focus:shadow-neon-sm appearance-none cursor-pointer"
          >
            {VALID_EMOTIONS.map(emo => (
              <option key={emo} value={emo}>{emo.toUpperCase()}</option>
            ))}
          </select>
        </div>
      </div>

      {/* ÁREA DE LA TABLA DE DATOS */}
      <div className="flex-1 bg-surface-1 border border-purple-800 relative overflow-hidden flex flex-col">
        {/* Glow de fondo */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-neon-purple/5 rounded-full blur-[100px] pointer-events-none"></div>

        {loading ? (
          <div className="flex-1 flex flex-col items-center justify-center">
            <div className="w-16 h-1 border-t-2 border-neon-purple animate-scan-fast mb-4"></div>
            <span className="font-mono text-purple-400 animate-pulse">QUERYING DATABASE...</span>
          </div>
        ) : error ? (
          <div className="flex-1 flex flex-col items-center justify-center text-red-500 font-mono">
            <AlertTriangle className="w-10 h-10 mb-2" />
            <p>{error}</p>
          </div>
        ) : records.length === 0 ? (
          <div className="flex-1 flex items-center justify-center font-mono text-purple-500">
            [ NO RECORDS FOUND FOR CURRENT FILTER ]
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-purple-900/50 font-mono text-[10px] text-purple-300 tracking-wider uppercase border-b border-purple-700">
                  <th className="p-4 w-24">ID_HASH</th>
                  <th className="p-4">TIMESTAMP_UTC</th>
                  <th className="p-4">DOMINANT_STATE</th>
                  <th className="p-4 w-1/3">MODEL_CONFIDENCE</th>
                  <th className="p-4">ENTROPY</th>
                </tr>
              </thead>
              <tbody className="font-mono text-sm text-purple-100">
                {records.map((record) => (
                  <tr key={record.id} className="border-b border-purple-900/50 hover:bg-purple-800/20 transition-colors">
                    <td className="p-4 text-purple-500 text-xs">{String(record.id).substring(0, 8)}...</td>
                    <td className="p-4">{formatCyberTime(record.timestamp)}</td>
                    <td className="p-4">
                      <span className="px-2 py-1 bg-purple-900 border border-purple-600 text-neon-purple text-xs tracking-wider">
                        {record.dominant_emotion.toUpperCase()}
                      </span>
                    </td>
                    <td className="p-4">
                      <div className="flex items-center gap-3">
                        <span className="text-purple-300 text-xs w-12">
                          {(record.confidence * 100).toFixed(1)}%
                        </span>
                        <div className="flex-1 h-1.5 bg-surface-3 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-neon-purple shadow-[0_0_8px_rgba(191,0,255,0.8)]"
                            style={{ width: `${record.confidence * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </td>
                    <td className="p-4">
                      <span className="text-purple-300 text-xs">
                        {record.entropy !== null && record.entropy !== undefined
                          ? Number(record.entropy).toFixed(3)
                          : '--'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* CONTROLES DE PAGINACIÓN */}
      <div className="flex items-center justify-between border-t border-purple-800 pt-4">
        <span className="font-mono text-xs text-purple-400">
          PAGE {page} OF {totalPages || 1}
        </span>
        
        <div className="flex gap-2">
          <button 
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1 || loading}
            className="p-2 border border-purple-700 bg-surface-1 text-purple-400 hover:bg-purple-800 hover:text-neon-purple disabled:opacity-30 disabled:cursor-not-allowed transition-all"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          
          <button 
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages || loading}
            className="p-2 border border-purple-700 bg-surface-1 text-purple-400 hover:bg-purple-800 hover:text-neon-purple disabled:opacity-30 disabled:cursor-not-allowed transition-all"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>
      </div>
      
    </div>
  );
};

export default History;
