import axios from "axios";

const API = "http://localhost:8000/api";

export const sendText = (user, text) =>
  axios.post(`${API}/text`, { user, text });

export const sendVoice = (user, audio) =>
  axios.post(`${API}/voice?user=${user}`, audio, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const getRequests = () =>
  axios.get(`${API}/requests`);
