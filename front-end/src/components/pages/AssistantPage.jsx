// src/pages/AssistantPage.jsx

import { useEffect, useState } from "react";
import TextInput from "../components/assistant/TextInput";
import VoiceRecorder from "../components/assistant/VoiceRecorder";
import DecisionCard from "../components/assistant/DecisionCard";
import RequestsHistory from "../components/assistant/RequestsHistory";
import QuickActions from "../components/assistant/QuickActions";

const API = "http://localhost:8000";

export default function AssistantPage() {
  const [users, setUsers] = useState({});
  const [currentUser, setCurrentUser] = useState(null);
  const [lastOutput, setLastOutput] = useState("");
  const [requests, setRequests] = useState([]);

  const loadState = async () => {
    const [usersRes, stateRes] = await Promise.all([
      fetch(`${API}/api/users`),
      fetch(`${API}/api/state`),
    ]);
    setUsers(await usersRes.json());
    const s = await stateRes.json();
    setCurrentUser(s.current_user);
    setLastOutput(s.last_visual);
    setRequests(s.recent_requests);
  };

  useEffect(() => {
    loadState();
  }, []);

  const sendText = async (text) => {
    const res = await fetch(`${API}/api/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    setCurrentUser(data.current_user);
    setLastOutput(data.visual);
    setRequests(data.recent_requests);
  };

  const sendVoice = async (blob) => {
    const form = new FormData();
    form.append("file", blob, "voice.webm");

    const res = await fetch(`${API}/api/voice`, { method: "POST", body: form });
    const data = await res.json();
    setCurrentUser(data.current_user);
    setLastOutput(data.visual);
    setRequests(data.recent_requests);
  };

  return (
    <div className="flex flex-col gap-6">
      <QuickActions onSelect={sendText} />

      <VoiceRecorder onSend={sendVoice} />

      <TextInput onSend={sendText} />

      <DecisionCard text={lastOutput} />

      <RequestsHistory items={requests} />
    </div>
  );
}
