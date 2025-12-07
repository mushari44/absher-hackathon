// src/App.jsx
import { useEffect, useState } from "react";
import "./styles/absher.css";

// Logos
import AbsherLogo from "./assets/absher.svg";
import MoiLogo from "./assets/moi.svg";
import Vision2030 from "./assets/vision2030.png";
import SaudiMan from "./assets/saudi_man.png";
import SaudiWoman from "./assets/saudi_woman.png";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [users, setUsers] = useState({});
  const [currentUser, setCurrentUser] = useState(null);
  const [currentUserKey, setCurrentUserKey] = useState("");

  const [lastVisual, setLastVisual] = useState("");
  const [recentRequests, setRecentRequests] = useState([]);

  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);

  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [voiceLoading, setVoiceLoading] = useState(false);

  // INITIAL LOAD
  useEffect(() => {
    const load = async () => {
      const usersRes = await fetch(`${API_BASE}/api/users`);
      const stateRes = await fetch(`${API_BASE}/api/state`);

      const usersData = await usersRes.json();
      const stateData = await stateRes.json();

      setUsers(usersData);
      setCurrentUser(stateData.current_user);
      setCurrentUserKey(stateData.current_user_key);
      setLastVisual(stateData.last_visual || "");
      setRecentRequests(stateData.recent_requests || []);
    };

    load();
  }, []);

  // SEND TEXT COMMAND
  const sendCommand = async () => {
    if (!text.trim()) return;

    setLoading(true);
    const res = await fetch(`${API_BASE}/api/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json();
    setCurrentUser(data.current_user);
    setLastVisual(data.visual);
    setRecentRequests(data.recent_requests || []);
    setText("");
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
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);

    setChunks([]);

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) setChunks((prev) => [...prev, e.data]);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: "audio/webm" });
      sendVoice(blob);
      stream.getTracks().forEach((t) => t.stop());
    };

    recorder.start();
    setMediaRecorder(recorder);
    setRecording(true);
  };

  // STOP RECORD
  const stopRecording = () => {
    mediaRecorder.stop();
    setRecording(false);
  };

  // SEND VOICE
  const sendVoice = async (blob) => {
    setVoiceLoading(true);

    const formData = new FormData();
    formData.append("file", blob, "voice.webm");

    const res = await fetch(`${API_BASE}/api/voice`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setCurrentUser(data.current_user);
    setLastVisual(data.visual);
    setRecentRequests(data.recent_requests || []);

    setVoiceLoading(false);
  };
const getUserAvatar = (user) => {
  if (!user) return SaudiMan;
  return user.gender === "female" ? SaudiWoman : SaudiMan;
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

          {/* TEXT INPUT CARD */}
          <div className="absher-card card-animate">
            <h2 className="card-title">📝 إدخال نصي</h2>
            <p className="card-desc">
              اكتب طلبك باللغة العربية أو الإنجليزية وسيقوم المساعد بتحديد الخدمة تلقائيًا.
            </p>

            <div className="text-row">
              <button className="absher-btn" onClick={sendCommand} disabled={loading}>
                {loading ? "جاري..." : "إرسال"}
              </button>

              <input
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="absher-input"
                placeholder="مثال: جدد الإقامة، كم المخالفات؟"
              />
            </div>
          </div>

          {/* VOICE CARD */}
          <div className="absher-card card-animate">
            <h2 className="card-title">🎤 التسجيل الصوتي المباشر</h2>
            <p className="card-desc">
              اضغط تسجيل وتحدث، وسيتم تحويل صوتك لنص وتحليل النية.
            </p>

            <button
              className={`absher-btn ${recording ? "absher-btn-danger pulse" : ""}`}
              onClick={recording ? stopRecording : startRecording}
            >
              {recording ? "إيقاف التسجيل" : "ابدأ التسجيل"}
            </button>

            {voiceLoading && <p className="loading-text">⏳ جارٍ معالجة الصوت...</p>}
          </div>

          {/* RESULT CARD */}
          <div className="absher-card card-animate">
            <h2 className="card-title">📌 النتيجة الأخيرة</h2>
            {lastVisual ? (
              <div
                className="absher-result"
                dangerouslySetInnerHTML={{ __html: lastVisual.replace(/\n/g, "<br/>") }}
              />
            ) : (
              <p className="card-desc">لم يتم تنفيذ أي أمر بعد.</p>
            )}
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
