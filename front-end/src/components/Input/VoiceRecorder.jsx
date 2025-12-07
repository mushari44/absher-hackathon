import { useState } from "react";

export default function VoiceRecorder({ onSend }) {
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [chunks, setChunks] = useState([]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    const recorder = new MediaRecorder(stream);
    setChunks([]);

    recorder.ondataavailable = (e) => setChunks((prev) => [...prev, e.data]);

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: "audio/webm" });
      onSend(blob);
    };

    recorder.start();
    setRecording(true);
    setMediaRecorder(recorder);
  };

  const stopRecording = () => {
    if (mediaRecorder) mediaRecorder.stop();
    setRecording(false);
  };

  return (
    <div className="bg-white border border-[#d5e4dd] p-5 rounded-xl">
      <h2 className="text-[#004d2a] text-lg font-semibold">🎤 التسجيل الصوتي</h2>

      <button
        onClick={recording ? stopRecording : startRecording}
        className={`mt-3 px-5 py-2 rounded-lg text-white font-semibold ${
          recording ? "bg-red-600" : "bg-[#006c3c]"
        }`}
      >
        {recording ? "إيقاف التسجيل" : "ابدأ التسجيل"}
      </button>
    </div>
  );
}
