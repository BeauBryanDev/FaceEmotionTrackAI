import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './context/AuthContext'

// Layouts
import AuthLayout from './layouts/AuthLayout'
import DashboardLayout from './layouts/DashboardLayout'

// Pages
import Login from './pages/Login'
import Register from './pages/Register'
import Dashboard from './pages/Dashboard'
import Home from './pages/Home'
import Emotions from './pages/Emotions'
import Profile from './pages/Profile'
import History from './pages/History'
import Inference from './pages/Inference'
import Analytics from './pages/Analytics'
import EmotionReport from './pages/EmotionReport'
import EmotionsAnalysis from './pages/EmotionsAnalysis'
import PCAAnalytics from './pages/PCAAnalytics'


// PROTECTED ROUTE
// Redirects to /login if user is not authenticated.
// Shows cyberpunk boot screen while auth state is loading.
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth()

  if (loading) return <CyberpunkLoader />
  if (!isAuthenticated) return <Navigate to="/login" replace />
  return children
}

// PUBLIC ROUTE

// Redirects authenticated users to /dashboard.
// Prevents logged-in users from seeing login/register pages.
const PublicRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth()

  if (loading) return <CyberpunkLoader />
  if (isAuthenticated) return <Navigate to="/about" replace />
  return children
}

// CYBERPUNK LOADER
// Shown while auth state is being restored from localStorage.

const CyberpunkLoader = () => (
  <div style={{
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'var(--purple-950)',
    gap: '24px',
  }}>
    <div style={{
      fontFamily: 'Orbitron, monospace',
      fontSize: '1.2rem',
      fontWeight: 900,
      color: 'var(--neon-purple)',
      textShadow: 'var(--glow-md)',
      letterSpacing: '0.3em',
      animation: 'pulse 1.5s ease-in-out infinite',
    }}>
      AUTHENTICATING
    </div>
    <div style={{
      width: '200px',
      height: '2px',
      background: 'rgba(170,0,255,0.2)',
      position: 'relative',
      overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute',
        height: '100%',
        width: '40%',
        background: 'linear-gradient(90deg, transparent, var(--neon-purple), transparent)',
        animation: 'scan 1.2s linear infinite',
      }} />
    </div>

    <style>{`
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
      }
      @keyframes scan {
        0%   { left: -40%; }
        100% { left: 140%; }
      }
    `}</style>
  </div>
)

// ROUTES

const App = () => (
  <Routes>
    {/* Public routes - redirect to dashboard if already logged in */}
    <Route element={<AuthLayout />}>
      <Route
        path="/login"
        element={<PublicRoute><Login /></PublicRoute>}
      />
      <Route
        path="/register"
        element={<PublicRoute><Register /></PublicRoute>}
      />
    </Route>

    {/* Protected routes - require authentication */}
    <Route
      element={
        <ProtectedRoute>
          <DashboardLayout />
        </ProtectedRoute>
      }
    >
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/inference" element={<Inference />} />
      <Route path="/history" element={<History />} />
      <Route path="/emotions" element={<Emotions />} />
      <Route path="/analytics" element={<Analytics />} />
      <Route path="/report" element={<EmotionReport />} />
      <Route path="/russelquadrants" element={<EmotionsAnalysis />} />
      <Route path="/pcaanalytics" element={<PCAAnalytics />} />
      <Route path="/profile" element={<Profile />} />
      <Route path="/about" element={<Home />} />
    </Route>

    {/* Default redirect */}
    <Route path="/" element={<Navigate to="/about" replace />} />
    <Route path="*" element={<Navigate to="/about" replace />} />
  </Routes>
)

export default App
