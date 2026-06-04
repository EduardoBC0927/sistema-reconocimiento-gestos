"use client";

import { useRef, useState } from "react";

export default function GesturesPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsCameraOn(true);
        setErrorMsg("");
      }
    } catch (error) {
      console.error("Error al acceder a la cámara:", error);
      setErrorMsg("No se pudo iniciar la cámara. Revisa los permisos en el navegador.");
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-950 p-8 text-white">
      <h1 className="mb-8 text-4xl font-bold tracking-tight text-blue-400">
        Reconocimiento de Gestos
      </h1>
      
      <div className="relative w-full max-w-3xl overflow-hidden rounded-2xl border-4 border-gray-800 bg-black shadow-2xl aspect-video flex items-center justify-center">
        
        {/* El video está oculto hasta que la cámara se encienda */}
        <video
          ref={videoRef}
          playsInline
          muted
          className={`h-full w-full object-cover transform -scale-x-100 ${isCameraOn ? 'block' : 'hidden'}`}
        ></video>

        {/* Botón para encender la cámara manualmente */}
        {!isCameraOn && (
          <button 
            onClick={startCamera}
            className="absolute z-10 px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-lg font-semibold transition-all shadow-lg"
          >
            Encender Cámara
          </button>
        )}
      </div>

      {errorMsg && (
        <p className="mt-4 text-red-400 font-medium">{errorMsg}</p>
      )}

      <p className="mt-6 text-gray-400">
        {isCameraOn ? "Cámara activa. Listo para reconocer gestos." : "Presiona el botón para comenzar."}
      </p>
    </main>
  );
}