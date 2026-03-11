import React from 'react';
import Text from '../components/ui/Text';
import emotitrackicon from '../assets/emotitrackincon.jpg';

const About = () => {
    return (
        <div className="p-4 sm:p-6 min-h-screen bg-black flex flex-col items-center gap-8 overflow-y-auto">
            {/* Container with neon borders */}
            <div className="w-full max-w-4xl border border-violet-800 bg-surface-0 p-4 sm:p-8 shadow-[0_0_15px_rgba(139,92,246,0.3)] relative group">

                {/* Neon Corners */}
                <div className="absolute top-0 left-0 w-8 h-8 border-t border-l border-neon-violet group-hover:shadow-[0_0_10px_#9d00ff] transition-all" />
                <div className="absolute top-0 right-0 w-8 h-8 border-t border-r border-neon-violet group-hover:shadow-[0_0_10px_#9d00ff] transition-all" />
                <div className="absolute bottom-0 left-0 w-8 h-8 border-b border-l border-neon-violet group-hover:shadow-[0_0_10px_#9d00ff] transition-all" />
                <div className="absolute bottom-0 right-0 w-8 h-8 border-b border-r border-neon-violet group-hover:shadow-[0_0_10px_#9d00ff] transition-all" />

                <div className="flex flex-col items-center mb-12">
                    {/* Main Brand Picture */}
                    <div className="mb-6 relative rounded-lg border-2 border-violet-500 shadow-[0_0_20px_rgba(139,92,246,0.7)] overflow-hidden">
                        <img
                            src={emotitrackicon}
                            alt="EMOTITRACK BRAND"
                            className="w-40 h-40 sm:w-48 sm:h-48 md:w-64 md:h-64 object-cover relative z-10 border border-purple-900 rounded-lg"
                            style={{ boxShadow: '0 0 20px rgba(170, 0, 255, 0.4)' }}
                        />
                    </div>

                    <Text variant="h1" glow className="text-3xl sm:text-5xl md:text-6xl text-center mb-2">
                        EMOTITRACK
                    </Text>
                    <Text variant="mono" className="text-purple-400 font-bold tracking-[0.4em] uppercase text-xs">
                        NEURAL_MONITORING_SUITE
                    </Text>
                </div>

                <div className="space-y-8 max-w-3xl mx-auto">
                    <section>
                        <Text variant="h3" className="mb-4 border-b border-purple-900 pb-2">Mission_Brief</Text>
                        <Text variant="body" className="leading-relaxed">
                            EMOTITRACK is a real-time biometric and emotion intelligence web application.
                            It captures a user’s live face stream, validates that the subject is a real person (anti-spoofing),
                            verifies identity against a registered facial template, detects the current emotion,
                            and stores historical emotion statistics for analytics.
                            Designed for authenticated users with a secure profile workflow and biometric-gated sensitive actions.
                        </Text>
                    </section>

                    <section>
                        <Text variant="h3" className="mb-4 border-b border-purple-900 pb-2">Operational_Capability</Text>
                        <ul className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <li className="data-readout bg-purple-950/30 p-3 border-l-2 border-neon-purple">
                                <Text variant="subtext" className="mb-1 text-purple-400">BIOMETRICS</Text>
                                <Text variant="mono" className="text-xs">Enrolls a 512D ArcFace embedding legacy per user node.</Text>
                            </li>
                            <li className="data-readout bg-purple-950/30 p-3 border-l-2 border-neon-purple">
                                <Text variant="subtext" className="mb-1 text-purple-400">LIVENESS</Text>
                                <Text variant="mono" className="text-xs">Anti-spoof protocols prevent replay attacks via MiniFASNetV2.</Text>
                            </li>
                            <li className="data-readout bg-purple-950/30 p-3 border-l-2 border-neon-purple">
                                <Text variant="subtext" className="mb-1 text-purple-400">EMOTION_ENG</Text>
                                <Text variant="mono" className="text-xs">EfficientNet-B0 backbone for real-time sentiment distribution.</Text>
                            </li>
                            <li className="data-readout bg-purple-950/30 p-3 border-l-2 border-neon-purple">
                                <Text variant="subtext" className="mb-1 text-purple-400">ANALYTICS</Text>
                                <Text variant="mono" className="text-xs">Recursive SVD/PCA projections of face-latent space mapping.</Text>
                            </li>
                        </ul>
                    </section>

                    <section>
                        <Text variant="h3" className="mb-4 border-b border-purple-900 pb-2">Core_Protocol_Stack</Text>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 sm:gap-8">
                            <div>
                                <Text variant="subtext" className="mb-2 text-purple-500">BACKEND</Text>
                                <Text variant="mono" className="text-[11px] leading-tight text-purple-300">
                                    FASTAPI // ONNX_RUNTIME<br />
                                    OPENCV // SQLALCHEMY<br />
                                    PGVECTOR // JWT_AUTH
                                </Text>
                            </div>
                            <div>
                                <Text variant="subtext" className="mb-2 text-purple-500">FRONTEND</Text>
                                <Text variant="mono" className="text-[11px] leading-tight text-purple-300">
                                    REACT_VITE // AXIOS<br />
                                    TAILWIND_CSS // RECHARTS<br />
                                    LUCIDE_ICONS
                                </Text>
                            </div>
                        </div>
                    </section>

                    <section>
                        <Text variant="h3" className="mb-4 border-b border-purple-900 pb-2">Data_Security</Text>
                        <Text variant="body" className="leading-relaxed italic text-purple-400">
                            User identity data, biometric templates (512D vectors), and emotion scores
                            are protected under secure hashing and pgvector encryption layers.
                            Sensitive profile updates require biometric verification when enrolled.
                        </Text>
                    </section>

                    {/* Footer decor */}
                    <div className="mt-12 pt-6 border-t border-purple-900/40 flex flex-col sm:flex-row sm:justify-between gap-2 text-[8px] font-mono text-purple-600 tracking-widest uppercase">
                        <span>Authored_By: BeauBryanDev</span>
                        <span>SYSTEM_STATUS: SECURE</span>
                        <span>BUILD: v1.0.4-PROTO</span>
                    </div>

                </div>
            </div>
        </div>
    );
};

export default About;
