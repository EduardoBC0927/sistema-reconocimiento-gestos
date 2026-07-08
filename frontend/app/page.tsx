"use client";

import { useRef, useState, useEffect } from "react";

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
        
        // Esperamos a que los metadatos del video estén listos antes de reproducir
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play()
            .then(() => {
              // Solo actualizamos la UI cuando el video ya se está reproduciendo
              setIsCameraOn(true);
              setErrorMsg("");
            })
            .catch((e) => {
              // Si React interrumpe la carga, ignoramos el error de tipo AbortError
              if (e.name !== 'AbortError') {
                console.error("Error al reproducir:", e);
                setErrorMsg("Error al reproducir el video.");
              }
            });
        };
      }
    } catch (error) {
      console.error("Error al acceder a la cámara:", error);
      setErrorMsg("No se pudo iniciar la cámara. Revisa los permisos.");
    }
  };

  // Buena práctica: Apagar la cámara si el usuario cambia de página
  useEffect(() => {
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-950 p-8 text-white">
      <h1 className="mb-8 text-4xl font-bold tracking-tight text-blue-400">
        Reconocimiento de Gestos
      </h1>
      
      <div className="relative w-full max-w-3xl overflow-hidden rounded-2xl border-4 border-gray-800 bg-black shadow-2xl aspect-video flex items-center justify-center">
        
        <video
          ref={videoRef}
          playsInline
          muted
          className={`h-full w-full object-cover transform -scale-x-100 ${isCameraOn ? 'block' : 'hidden'}`}
        ></video>

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