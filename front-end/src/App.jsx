// src/App.jsx
import { useEffect, useState } from "react";
import "./styles/absher.css";

// Logos
import AbsherLogo from "./assets/absher.svg";
import MoiLogo from "./assets/moi.svg";
import Vision2030 from "./assets/vision2030.png";
import SaudiMan from "./assets/saudi_man.png";
import SaudiWoman from "./assets/saudi_woman.png";
import RobotMan from "./assets/robot_man.png";
import RobotWoman from "./assets/robot_woman.png";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [users, setUsers] = useState({});
  const [currentUser, setCurrentUser] = useState(null);
  const [currentUserKey, setCurrentUserKey] = useState("");

  const [recentRequests, setRecentRequests] = useState([]);
  const [messages, setMessages] = useState([]); // ChatGPT-like conversation history

  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);

  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);

  // INITIAL LOAD
  useEffect(() => {
    const load = async () => {
      const usersRes = await fetch(`${API_BASE}/api/users`);
      const stateRes = await fetch(`${API_BASE}/api/state`);

      const usersData = await usersRes.json();
      const stateData = await stateRes.json();

      setUsers(usersData);
      setCurrentUser(usersData[stateData.current_user_key]);
      setCurrentUserKey(stateData.current_user_key);
      setRecentRequests(stateData.recent_requests || []);
    };

    load();
  }, []);

  // SEND TEXT COMMAND
  const sendCommand = async () => {
    if (!text.trim()) return;

    // Add user message to chat
    const userMessage = { type: "user", text: text };
    setMessages((prev) => [...prev, userMessage]);

    const userInput = text;
    setText("");
    setLoading(true);

    const res = await fetch(`${API_BASE}/api/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: userInput }),
    });

    const data = await res.json();
    setCurrentUser(data.current_user);
    setRecentRequests(data.recent_requests || []);

    // Add assistant response to chat
    const assistantMessage = {
      type: "assistant",
      text: data.visual,
      steps: data.action_steps,
      intent: data.intent,
    };
    setMessages((prev) => [...prev, assistantMessage]);
    setLoading(false);
  };

  // SWITCH USER
  const switchUser = async (key) => {
    const res = await fetch(`${API_BASE}/api/switch-user`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_key: key }),
    });

    const data = await res.json();
    setCurrentUser(data.current_user);
    setCurrentUserKey(key);
  };

  // START VOICE RECORD
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Set recording state first
      setRecording(true);

      // Start recorder
      let localChunks = [];
      const recorder = new MediaRecorder(stream);

      recorder.ondataavailable = e => localChunks.push(e.data);
      recorder.onstop = () => {
        const blob = new Blob(localChunks, { type: "audio/webm" });
        sendVoice(blob);
        stream.getTracks().forEach(t => t.stop());
      };

      recorder.start(200);
      setMediaRecorder(recorder);

      // Setup waveform visualization after state is set
      setTimeout(() => {
        const canvas = document.getElementById("waveform");
        if (canvas) {
          const audioCtx = new AudioContext();
          const source = audioCtx.createMediaStreamSource(stream);
          const analyser = audioCtx.createAnalyser();
          analyser.fftSize = 256;
          source.connect(analyser);

          const ctx = canvas.getContext("2d");

          function drawWave() {
            if (!recorder || recorder.state === "inactive") return;
            requestAnimationFrame(drawWave);

            let dataArray = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(dataArray);

            ctx.fillStyle = "transparent";
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            ctx.strokeStyle = "#0A8754";
            ctx.lineWidth = 2;
            ctx.beginPath();

            let sliceWidth = canvas.width / dataArray.length;
            let x = 0;

            for (let i = 0; i < dataArray.length; i++) {
              let v = dataArray[i] / 255.0;
              let y = (canvas.height / 2) - (v * 15);

              i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
              x += sliceWidth;
            }

            ctx.stroke();
          }

          drawWave();
        }
      }, 100);
    } catch (error) {
      console.error("Error starting recording:", error);
      alert("تعذر الوصول إلى الميكروفون. يرجى السماح بالوصول للميكروفون.");
    }
  };


  // STOP RECORD
  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    setRecording(false);
  };

  // SEND VOICE
  const sendVoice = async (blob) => {
    setLoading(true);

    const formData = new FormData();
    formData.append("file", blob, "voice.webm");

    const res = await fetch(`${API_BASE}/api/voice`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setCurrentUser(data.current_user);
    setRecentRequests(data.recent_requests || []);

    // Add user voice message
    const userMessage = { type: "user", text: data.text, isVoice: true };
    setMessages((prev) => [...prev, userMessage]);

    // Add assistant response
    const assistantMessage = {
      type: "assistant",
      text: data.visual,
      steps: data.action_steps,
      intent: data.intent,
    };
    setMessages((prev) => [...prev, assistantMessage]);

    setLoading(false);
  };
const getUserAvatar = (user) => {
  if (!user) return SaudiMan;
  return user.gender === "female" ? SaudiWoman : SaudiMan;
};

const getBotAvatar = (user) => {
  if (!user) return RobotMan;
  return user.gender === "female" ? RobotWoman : RobotMan;
};

  return (
    <div className="absher-app fade-in">

      {/* ================= HEADER ================= */}
      <header className="absher-header">
        <img src={MoiLogo} className="gov-logo" alt="MOI" />
        <div className="absher-center-header">
          <img src={AbsherLogo} className="absher-main-logo" alt="Absher" />
          <h2>أبشر مساعد للخدمات الرقمية</h2>
        </div>
        <img src={Vision2030} className="gov-logo" alt="Vision 2030" />
      </header>

      {/* ================= MAIN LAYOUT ================= */}
      <div className="absher-layout">

        {/* ============== SIDEBAR ============== */}
        <aside className="absher-sidebar absher-shadow slide-right">
          <h3 className="sidebar-title">اختر المستخدم</h3>

          <div className="sidebar-users">
            {Object.entries(users).map(([key, user]) => (
              <button
                key={key}
                onClick={() => switchUser(key)}
                className={
                  "sidebar-user-btn " +
                  (currentUserKey === key ? "active" : "")
                }
              >
<div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
  <img 
    src={getUserAvatar(user)}
    alt="avatar"
    style={{ width: "42px", height: "42px", borderRadius: "50%" }}
  />
  <div>
    <div className="name">{user.name}</div>
    <div className="type">{user.user_type}</div>
  </div>
</div>

              </button>
            ))}
          </div>

          <div className="sidebar-hint">
            <p>جرّب أوامر:</p>
            <ul>
              <li>جدد رخصتي</li>
              <li>كم باقي على الإقامة؟</li>
              <li>change user to alex</li>
            </ul>
          </div>
        </aside>

        {/* ============== MAIN CONTENT ============== */}
        <main className="absher-main">

          {/* CHAT INTERFACE - ChatGPT Style */}
          <div className="chat-container card-animate">
            <div className="chat-header">
              <h2 className="card-title">💬 مساعد أبشر الذكي</h2>
              <p className="card-desc">اكتب أو تحدث لتنفيذ خدماتك</p>
            </div>

            {/* Messages Area */}
            <div className="chat-messages">
              {messages.length === 0 ? (
                <div className="chat-empty">
                  <div className="empty-icon">
                    <img src={getBotAvatar(currentUser)} alt="Robot" style={{ width: "80px", height: "80px", borderRadius: "50%" }} />
                  </div>
                  <p>مرحباً! كيف يمكنني مساعدتك اليوم؟</p>
                  <div className="suggestions">
                    <button onClick={() => setText("جدد رخصتي")} className="suggestion-btn">
                      جدد رخصتي
                    </button>
                    <button onClick={() => setText("كم باقي على الإقامة؟")} className="suggestion-btn">
                      كم باقي على الإقامة؟
                    </button>
                    <button onClick={() => setText("أبغى موعد جوازات")} className="suggestion-btn">
                      أبغى موعد جوازات
                    </button>
                  </div>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div key={idx} className={`chat-message ${msg.type}`}>
                    <div className="message-avatar">
                      {msg.type === "user" ? (
                        <img src={getUserAvatar(currentUser)} alt="User" />
                      ) : (
                        <img src={getBotAvatar(currentUser)} alt="Robot" />
                      )}
                    </div>
                    <div className="message-content">
                      {msg.isVoice && <span className="voice-badge">🎤 صوتي</span>}
                      <div className="message-text">{msg.text}</div>
                      {msg.steps && (
                        <div className="message-steps">
                          <strong>📋 خطوات التنفيذ:</strong>
                          <div dangerouslySetInnerHTML={{ __html: msg.steps.replace(/\n/g, "<br/>") }} />
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
              {loading && (
                <div className="chat-message assistant">
                  <div className="message-avatar">
                    <img src={getBotAvatar(currentUser)} alt="Robot" />
                  </div>
                  <div className="message-content">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Input Area */}
            <div className="chat-input-area">
              {recording && (
                <div className="recording-indicator">
                  <div className="rec-dot"></div>
                  <span>جارٍ التسجيل...</span>
                  <canvas id="waveform" className="waveform-mini"></canvas>
                </div>
              )}
              <div className="chat-input-row">
                <button
                  className={`voice-btn ${recording ? "recording" : ""}`}
                  onClick={recording ? stopRecording : startRecording}
                  disabled={loading}
                >
                  <i className="fas fa-microphone"></i>
                </button>

                <input
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendCommand()}
                  className="chat-input"
                  placeholder="اكتب رسالتك هنا..."
                  disabled={loading || recording}
                />

                <button
                  className="send-btn"
                  onClick={sendCommand}
                  disabled={loading || !text.trim() || recording}
                >
                  <i className="fas fa-paper-plane"></i>
                </button>
              </div>
            </div>
          </div>

          {/* REQUESTS CARD */}
          <div className="absher-card card-animate">
            <h2 className="card-title">🗂️ آخر الطلبات</h2>

            {recentRequests.length ? (
              <div className="requests-list">
                {recentRequests.map((req) => (
                  <div key={req.request_id} className="request-item">
                    <div className="request-main">
                      <span>رقم: {req.request_id}</span>
                      <span>الحالة: {req.status}</span>
                    </div>
                    <div className="request-meta">الخدمة: {req.service_id}</div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="card-desc">لا توجد طلبات حتى الآن.</p>
            )}
          </div>

        </main>
      </div>
    </div>
  );
}
