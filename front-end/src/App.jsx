// src/App.jsx
import { useEffect, useState, useRef } from "react";
import "./styles/absher.css";

// Logos
import AbsherLogo from "./assets/absher.svg";
import MoiLogo from "./assets/moi.svg";
import Vision2030 from "./assets/vision2030.png";
import SaudiMan from "./assets/saudi_man.png";
import SaudiWoman from "./assets/saudi_woman.png";
import RobotMan from "./assets/robot_man.png";
import RobotWoman from "./assets/robot_woman.png";

// API Base URL - Update this for production
const API_BASE = "http://localhost:8000";
export default function App() {
  const [users, setUsers] = useState({});
  const [currentUser, setCurrentUser] = useState(null);
  const [currentUserKey, setCurrentUserKey] = useState("");

  const [recentRequests, setRecentRequests] = useState([]);

  // Store messages per user: { userKey: [messages] }
  const [messagesByUser, setMessagesByUser] = useState({});

  // Current user's messages (derived from messagesByUser)
  const messages = messagesByUser[currentUserKey] || [];

  // Helper function to update messages for the current user
  const setCurrentUserMessages = (updater) => {
    setMessagesByUser(prev => ({
      ...prev,
      [currentUserKey]: typeof updater === 'function'
        ? updater(prev[currentUserKey] || [])
        : updater
    }));
  };

  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);

  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const currentAudioRef = useRef(null); // Track currently playing audio
  const audioUrlsRef = useRef([]); // Track audio URLs for cleanup
  const chunksRef = useRef([]); // Use ref to avoid stale closure
  const streamRef = useRef(null); // Track audio stream for cleanup
  const animationFrameRef = useRef(null); // Track animation frame for cleanup
  const audioCtxRef = useRef(null); // Track AudioContext for cleanup
  const stopTimeoutRef = useRef(null); // Auto-stop timeout
  const messagesEndRef = useRef(null); // Scroll to bottom ref
  const isStartingRef = useRef(false); // Prevent double-start

  useEffect(() => {
    const load = async () => {
      try {
        const usersRes = await fetch(`${API_BASE}/api/users`, {
          headers: {
            "ngrok-skip-browser-warning": "true",
          },
        });

        const usersData = await usersRes.json();
        console.log("usersData:", usersData);

        const stateRes = await fetch(`${API_BASE}/api/state`, {
          headers: {
            "ngrok-skip-browser-warning": "true",
          },
        });

        const stateData = await stateRes.json();
        console.log("stateData:", stateData);

        const initialUserKey = stateData.current_user_key;

        setUsers(usersData);
        setCurrentUser(usersData[initialUserKey]);
        setRecentRequests(stateData.recent_requests || []);

        // Fetch notification and set it in messagesByUser BEFORE setting currentUserKey
        const notificationRes = await fetch(`${API_BASE}/api/notification/${initialUserKey}`, {
          headers: {
            "ngrok-skip-browser-warning": "true",
          },
        });
        const notificationData = await notificationRes.json();

        if (!notificationData.error) {
          const welcomeMessage = {
            type: "assistant",
            text: notificationData.notification,
            isWelcome: true,
            userKey: initialUserKey,
          };

          // Set initial messages for the initial user
          setMessagesByUser({
            [initialUserKey]: [welcomeMessage]
          });
        }

        // Set currentUserKey AFTER setting messages
        setCurrentUserKey(initialUserKey);
      } catch (err) {
        console.error("LOAD ERROR:", err);
      }
    };

    load();
  }, []);

  // SEND TEXT COMMAND
  const sendCommand = async () => {
    if (!text.trim() || loading) return;

    // Add user message to chat
    const userMessage = { type: "user", text: text, userKey: currentUserKey };
    setCurrentUserMessages((prev) => [...prev, userMessage]);

    const userInput = text;
    setText("");
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/command`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userInput, user_key: currentUserKey }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();
      setCurrentUser(data.current_user);
      setRecentRequests(data.recent_requests || []);

      // Add assistant response to chat
      const assistantMessage = {
        type: "assistant",
        text: data.visual,
        steps: data.action_steps,
        intent: data.intent,
        userKey: currentUserKey,
      };
      setCurrentUserMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("sendCommand error:", error);
      setCurrentUserMessages((prev) => [
        ...prev,
        { type: "assistant", text: "تعذر تنفيذ الطلب الآن، حاول مرة أخرى لاحقاً.", error: true },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // SWITCH USER
  const switchUser = async (key) => {
    // Don't switch if already on this user
    if (key === currentUserKey) {
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/api/switch-user`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_key: key }),
      });

      const data = await res.json();
      if (data.error) {
        // Use old currentUserKey since we haven't switched yet
        const errorMessage = {
          type: "assistant",
          text: `تعذر تغيير المستخدم: ${data.error}`,
          error: true
        };
        setMessagesByUser(prev => ({
          ...prev,
          [currentUserKey]: [...(prev[currentUserKey] || []), errorMessage]
        }));
        return;
      }

      setCurrentUser(data.current_user);
      setCurrentUserKey(key);

      // Only fetch notification if this user doesn't have messages yet
      if (!messagesByUser[key] || messagesByUser[key].length === 0) {
        // Fetch notification for new user
        const notificationRes = await fetch(`${API_BASE}/api/notification/${key}`, {
          headers: {
            "ngrok-skip-browser-warning": "true",
          },
        });
        const notificationData = await notificationRes.json();

        if (!notificationData.error) {
          const welcomeMessage = {
            type: "assistant",
            text: notificationData.notification,
            isWelcome: true,
            userKey: key,
          };

          setMessagesByUser(prev => ({
            ...prev,
            [key]: [welcomeMessage]
          }));
        }
      }
    } catch (error) {
      console.error("switchUser error:", error);
      const errorMessage = {
        type: "assistant",
        text: "تعذر تغيير المستخدم حالياً.",
        error: true
      };
      setMessagesByUser(prev => ({
        ...prev,
        [currentUserKey]: [...(prev[currentUserKey] || []), errorMessage]
      }));
    }
  };

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Cleanup on unmount ONLY (not on mediaRecorder change!)
  useEffect(() => {
    return () => {
      console.log('🧹 Component unmounting - cleaning up');

      // Stop recording if active - use ref to get latest value
      const recorder = mediaRecorder;
      if (recorder && recorder.state !== "inactive") {
        console.log('🛑 Stopping recorder on unmount');
        recorder.stop();
      }
      // Stop audio stream
      const stream = streamRef.current;
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
      // Stop audio playback
      const audio = currentAudioRef.current;
      if (audio) {
        audio.pause();
      }
      // Revoke all audio URLs to prevent memory leaks
      audioUrlsRef.current.forEach(url => URL.revokeObjectURL(url));
      audioUrlsRef.current = [];
      // Cancel animation frame
      const animFrame = animationFrameRef.current;
      if (animFrame) {
        cancelAnimationFrame(animFrame);
      }
      // Close audio context
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {});
        audioCtxRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run ONLY on mount/unmount, not on mediaRecorder changes!


  // START VOICE RECORD
  const startRecording = async () => {
    // Prevent double-triggering
    if (recording || mediaRecorder || isStartingRef.current) {
      console.log('⚠️ Recording already in progress, ignoring duplicate start');
      return;
    }

    isStartingRef.current = true;
    console.log('🎙️ Start recording requested');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000
        }
      });

      // Store stream for cleanup
      streamRef.current = stream;

      // Reset chunks array BEFORE starting
      chunksRef.current = [];

      // Start recorder BEFORE setting state (critical for waveform timing)
      // Try multiple codecs for best compatibility
      let mimeType = 'audio/webm';
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        mimeType = 'audio/webm;codecs=opus';
        console.log('✅ Using codec: audio/webm;codecs=opus');
      } else if (MediaRecorder.isTypeSupported('audio/webm')) {
        mimeType = 'audio/webm';
        console.log('⚠️ Falling back to: audio/webm');
      } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
        mimeType = 'audio/ogg;codecs=opus';
        console.log('⚠️ Falling back to: audio/ogg;codecs=opus');
      } else {
        console.warn('⚠️ No preferred codec supported, using default');
      }

      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
          console.log(`📦 Chunk received: ${e.data.size} bytes. Total chunks: ${chunksRef.current.length}`);
        }
      };

      recorder.onstop = () => {
        console.log(`🛑 Recorder stopped. Total chunks: ${chunksRef.current.length}`);
        console.log(`📊 Recorder final state: ${recorder.state}`);

        // Create blob from chunks
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        console.log(`🎵 Blob created: ${blob.size} bytes`);

        // Debug: Check why we got 0 chunks
        if (chunksRef.current.length === 0) {
          console.error('❌ ZERO chunks collected! This means ondataavailable never fired or stream stopped immediately.');
        }

        // Validate blob size
        if (blob.size < 500) {
          setCurrentUserMessages((prev) => [
            ...prev,
            { type: "assistant", text: "التسجيل قصير جداً. الرجاء المحاولة مرة أخرى.", error: true },
          ]);

          // Cleanup after error
          if (streamRef.current) {
            streamRef.current.getTracks().forEach((t) => t.stop());
            streamRef.current = null;
          }
          if (audioCtxRef.current) {
            audioCtxRef.current.close().catch(() => {});
            audioCtxRef.current = null;
          }
          if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
          }
          setMediaRecorder(null);
          setRecording(false);
          chunksRef.current = [];
          return;
        }

        // Send valid blob
        sendVoice(blob);

        // Cleanup after successful recording
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
        }
        if (audioCtxRef.current) {
          audioCtxRef.current.close().catch(() => {});
          audioCtxRef.current = null;
        }
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
        setMediaRecorder(null);
        setRecording(false);
        chunksRef.current = [];
      };

      recorder.onerror = (e) => {
        console.error('❌ MediaRecorder error:', e);
        setCurrentUserMessages((prev) => [
          ...prev,
          { type: "assistant", text: "حدث خطأ في التسجيل. الرجاء المحاولة مرة أخرى.", error: true },
        ]);

        // Force cleanup
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
        }
        if (audioCtxRef.current) {
          audioCtxRef.current.close().catch(() => {});
          audioCtxRef.current = null;
        }
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
        setMediaRecorder(null);
        setRecording(false);
        chunksRef.current = [];
      };

      // Store recorder BEFORE starting (critical!)
      setMediaRecorder(recorder);
      setRecording(true);

      // Start recording with small timeslices for better data flow
      try {
        recorder.start(200);
        console.log('🎤 Recording started');
        console.log(`📊 Recorder state: ${recorder.state}`);
        console.log(`📊 Stream active: ${stream.active}`);
        console.log(`📊 Stream tracks: ${stream.getTracks().length}`);

        // Verify recorder is actually recording
        setTimeout(() => {
          if (recorder.state !== "recording") {
            console.error(`❌ Recorder state is ${recorder.state} after start - should be "recording"!`);
            setCurrentUserMessages((prev) => [
              ...prev,
              { type: "assistant", text: "فشل بدء التسجيل. تأكد من السماح بالوصول للميكروفون.", error: true },
            ]);
            // Cleanup
            if (streamRef.current) {
              streamRef.current.getTracks().forEach((t) => t.stop());
              streamRef.current = null;
            }
            setMediaRecorder(null);
            setRecording(false);
            isStartingRef.current = false;
          }
        }, 100);
      } catch (startError) {
        console.error('❌ recorder.start() failed:', startError);
        setCurrentUserMessages((prev) => [
          ...prev,
          { type: "assistant", text: `فشل بدء التسجيل: ${startError.message}`, error: true },
        ]);
        // Cleanup
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
        }
        setMediaRecorder(null);
        setRecording(false);
        isStartingRef.current = false;
        return;
      }

      // Clear the starting flag after a short delay
      setTimeout(() => {
        isStartingRef.current = false;
        console.log(`✅ isStartingRef cleared. Recorder state: ${recorder.state}`);
      }, 500);

      // Safety auto-stop after 15 seconds
      stopTimeoutRef.current = setTimeout(() => {
        console.log('⏰ Auto-stop timeout triggered');
        if (recorder && recorder.state === "recording") {
          recorder.stop();
        }
      }, 15000);

      // Setup waveform visualization after a small delay to ensure recorder is stable
      setTimeout(() => {
        // Only setup if recorder is still recording
        if (!recorder || recorder.state !== "recording") {
          console.log('⚠️ Recorder not recording, skipping waveform setup');
          return;
        }

        const canvas = document.getElementById("waveform");
        if (canvas) {
          try {
            // @ts-ignore - AudioContext fallback for older browsers
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            audioCtxRef.current = audioCtx;
            const source = audioCtx.createMediaStreamSource(stream);
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.8;
            source.connect(analyser);
            console.log('✅ Waveform visualization setup complete');

            const ctx = canvas.getContext("2d");
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);

            function drawWave() {
              // Check if recorder is still active
              if (!recorder || recorder.state === "inactive") {
                if (animationFrameRef.current) {
                  cancelAnimationFrame(animationFrameRef.current);
                  animationFrameRef.current = null;
                }
                // Clear canvas on stop
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                return;
              }

              animationFrameRef.current = requestAnimationFrame(drawWave);

              analyser.getByteFrequencyData(dataArray);

              ctx.fillStyle = "rgba(255, 255, 255, 0.1)";
              ctx.fillRect(0, 0, canvas.width, canvas.height);

              ctx.strokeStyle = "#0A8754";
              ctx.lineWidth = 2;
              ctx.beginPath();

              const sliceWidth = canvas.width / bufferLength;
              let x = 0;

              for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 255.0;
                const y = canvas.height / 2 - v * 20;

                if (i === 0) {
                  ctx.moveTo(x, y);
                } else {
                  ctx.lineTo(x, y);
                }
                x += sliceWidth;
              }

              ctx.stroke();
            }

            drawWave();
          } catch (waveformError) {
            console.error('❌ Waveform setup error:', waveformError);
            // Continue recording even if waveform fails
          }
        }
      }, 200);
    } catch (error) {
      console.error("❌ Error starting recording:", error);
      setCurrentUserMessages((prev) => [
        ...prev,
        { type: "assistant", text: "تعذر الوصول إلى الميكروفون. يرجى السماح بالوصول للميكروفون.", error: true },
      ]);
      setRecording(false);
      isStartingRef.current = false; // Reset on error
    }
  };

  // STOP RECORD
  const stopRecording = () => {
    console.log('⏹️ Stop recording called');

    // Don't stop if we're still starting
    if (isStartingRef.current) {
      console.log('⚠️ Still starting recording, ignoring stop request');
      return;
    }

    // Don't stop if not recording
    if (!recording && (!mediaRecorder || mediaRecorder.state === "inactive")) {
      console.log('⚠️ Not recording, ignoring stop request');
      return;
    }

    // Clear auto-stop timeout
    if (stopTimeoutRef.current) {
      clearTimeout(stopTimeoutRef.current);
      stopTimeoutRef.current = null;
    }

    // Stop the recorder (this will trigger onstop handler)
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      console.log('🛑 Stopping MediaRecorder...');
      mediaRecorder.stop();
    }

    // Stop stream tracks immediately
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    // Note: DON'T clear chunksRef here - onstop handler needs it!
    // Note: DON'T close audioContext here - animation loop will handle it
    // Note: DON'T cancel animation frame - drawWave checks recorder.state
  };

  // SEND VOICE
  const sendVoice = async (blob) => {
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", blob, "voice.webm");
      formData.append("user_key", currentUserKey);

      const res = await fetch(`${API_BASE}/api/voice`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();

      // Check for errors
      if (data.error) {
        console.error("Voice processing error:", data.error);
        const errorMessage = {
          type: "assistant",
          text: "عذراً، حدث خطأ في معالجة الصوت. الرجاء المحاولة مرة أخرى.",
        };
        setCurrentUserMessages((prev) => [...prev, errorMessage]);
        setLoading(false);
        return;
      }

      setCurrentUser(data.current_user);
      setRecentRequests(data.recent_requests || []);

      // Add user voice message
      const userMessage = { type: "user", text: data.text, isVoice: true, userKey: currentUserKey };
      setCurrentUserMessages((prev) => [...prev, userMessage]);

      // Add assistant response with audio
      const assistantMessage = {
        type: "assistant",
        text: data.visual,
        steps: data.action_steps,
        intent: data.intent,
        audio: data.audio, // Base64 encoded audio
        audioFormat: data.audio_format,
        userKey: currentUserKey,
      };
      setCurrentUserMessages((prev) => [...prev, assistantMessage]);

      // Play audio response automatically
      if (data.audio) {
        playAudio(data.audio);
      }
    } catch (error) {
      console.error("Error sending voice:", error);
      const errorMessage = {
        type: "assistant",
        text: "عذراً، حدث خطأ في الاتصال. الرجاء التحقق من اتصال الإنترنت.",
      };
      setCurrentUserMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  // PLAY AUDIO FROM BASE64
  const playAudio = (base64Audio) => {
    try {
      // Stop any currently playing audio
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current.currentTime = 0;
        currentAudioRef.current = null;
      }

      // Convert base64 to blob
      const byteCharacters = atob(base64Audio);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: "audio/mpeg" });
      const audioUrl = URL.createObjectURL(blob);

      // Track URL for cleanup
      audioUrlsRef.current.push(audioUrl);

      // Create and play new audio
      const audio = new Audio(audioUrl);
      currentAudioRef.current = audio;

      audio.play().catch(err => {
        console.error("❌ Audio playback failed:", err);
        URL.revokeObjectURL(audioUrl);
        currentAudioRef.current = null;
      });

      // Cleanup when audio ends
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        // Remove from tracking array
        const idx = audioUrlsRef.current.indexOf(audioUrl);
        if (idx > -1) audioUrlsRef.current.splice(idx, 1);
        currentAudioRef.current = null;
      };

      // Cleanup if audio errors
      audio.onerror = (err) => {
        console.error("❌ Audio error:", err);
        URL.revokeObjectURL(audioUrl);
        // Remove from tracking array
        const idx = audioUrlsRef.current.indexOf(audioUrl);
        if (idx > -1) audioUrlsRef.current.splice(idx, 1);
        currentAudioRef.current = null;
      };
    } catch (error) {
      console.error("❌ Error playing audio:", error);
    }
  };
  const getUserAvatar = (user) => {
    if (!user) return SaudiMan;
    return user.gender === "female" ? SaudiWoman : SaudiMan;
  };

  const getBotAvatar = (user) => {
    if (!user) return RobotMan;
    return user.gender === "female" ? RobotWoman : RobotMan;
  };

  // UPLOAD PHOTO
  const uploadPhoto = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/api/upload-photo`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (data.success) {
        // Add success message to chat
        const successMessage = {
          type: "assistant",
          text: `${data.message}\n\nاسم الملف: ${data.file_name}\nالحجم: ${(data.file_size / 1024).toFixed(2)} كيلوبايت\nتاريخ الرفع: ${data.upload_date}`,
        };
        setCurrentUserMessages((prev) => [...prev, successMessage]);
      } else {
        // Add error message to chat
        const errorMessage = {
          type: "assistant",
          text: `❌ ${data.error}`,
        };
        setCurrentUserMessages((prev) => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error("Photo upload error:", error);
      const errorMessage = {
        type: "assistant",
        text: "عذراً، حدث خطأ أثناء رفع الصورة. الرجاء المحاولة مرة أخرى.",
      };
      setCurrentUserMessages((prev) => [...prev, errorMessage]);
    }

    setLoading(false);
    // Reset file input
    event.target.value = "";
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
                  "sidebar-user-btn " + (currentUserKey === key ? "active" : "")
                }
              >
                <div
                  style={{ display: "flex", alignItems: "center", gap: "12px" }}
                >
                  <img
                    src={getUserAvatar(user)}
                    alt="avatar"
                    className="sidebar-user-avatar"
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
                    <img
                      src={getBotAvatar(currentUser)}
                      alt="Robot"
                      className="empty-robot"
                    />
                  </div>
                  <p>مرحباً! كيف يمكنني مساعدتك اليوم؟</p>
                  <div className="suggestions">
                    <button
                      onClick={() => setText("جدد رخصتي")}
                      className="suggestion-btn"
                    >
                      جدد رخصتي
                    </button>
                    <button
                      onClick={() => setText("كم باقي على الإقامة؟")}
                      className="suggestion-btn"
                    >
                      كم باقي على الإقامة؟
                    </button>
                    <button
                      onClick={() => setText("أبغى موعد جوازات")}
                      className="suggestion-btn"
                    >
                      أبغى موعد جوازات
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  {messages.map((msg, idx) => (
                    <div key={idx} className={`chat-message ${msg.type}`}>
                      <div className="message-avatar">
                        {(() => {
                          const messageUser = users[msg.userKey] || currentUser;
                          if (msg.type === "user") {
                            return <img src={getUserAvatar(messageUser)} alt="User" />;
                          }
                          return <img src={getBotAvatar(messageUser)} alt="Robot" />;
                        })()}
                      </div>
                      <div className="message-content">
                        {msg.isVoice && (
                          <span className="voice-badge">🎤 صوتي</span>
                        )}
                        <div className="message-text">{msg.text}</div>
                        {msg.steps && (
                          <div className="message-steps">
                            <strong>📋 خطوات التنفيذ:</strong>
                            <div
                              dangerouslySetInnerHTML={{
                                __html: msg.steps.replace(/\n/g, "<br/>"),
                              }}
                            />
                          </div>
                        )}
                        {msg.audio && (
                          <button
                            className="replay-audio-btn"
                            onClick={() => playAudio(msg.audio)}
                            title="إعادة تشغيل الصوت"
                          >
                            <i className="fas fa-volume-up"></i> إعادة تشغيل الصوت
                          </button>
                        )}
                      </div>
                    </div>
                  ))}

                  {/* Invisible div for auto-scroll */}
                  <div ref={messagesEndRef} />

                  {/* Show suggestions after welcome message */}
                  {messages.length === 1 && messages[0].isWelcome && (
                    <div className="suggestions-inline">
                      <p className="suggestions-title">جرّب الأوامر التالية:</p>
                      <div className="suggestions">
                        <button
                          onClick={() => setText("جدد رخصتي")}
                          className="suggestion-btn"
                        >
                          جدد رخصتي
                        </button>
                        <button
                          onClick={() => setText("كم باقي على الإقامة؟")}
                          className="suggestion-btn"
                        >
                          كم باقي على الإقامة؟
                        </button>
                        <button
                          onClick={() => setText("أبغى موعد جوازات")}
                          className="suggestion-btn"
                        >
                          أبغى موعد جوازات
                        </button>
                      </div>
                    </div>
                  )}
                </>
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

                <label className="upload-photo-btn" title="رفع صورة شخصية">
                  <i className="fas fa-camera"></i>
                  <input
                    type="file"
                    accept="image/jpeg,image/jpg,image/png"
                    onChange={uploadPhoto}
                    style={{ display: "none" }}
                    disabled={loading}
                  />
                </label>

                <input
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) =>
                    e.key === "Enter" && !e.shiftKey && sendCommand()
                  }
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
                    <div className="request-meta">
                      الخدمة: {req.service_id}
                      {req.service_id === "5001" && req.plate && (
                        <>
                          <br />
                          <small>
                            🚗 اللوحة: {req.plate} |
                            من: {req.from_user} → إلى: {req.to_user} |
                            السعر: {req.price} ریال
                          </small>
                          {req.timestamp && (
                            <>
                              <br />
                              <small>📅 {req.timestamp}</small>
                            </>
                          )}
                        </>
                      )}
                    </div>
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
