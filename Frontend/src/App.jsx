import { useState } from 'react';
import Navbar from './compo/Navbar';
import OldChat from './compo/OldChat';
import ChatBox from './compo/ChatBox';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hello! I am CampusGPT. Ask me anything' },
  ]);

  const handleSend = async (message) => {
    if (!message.trim()) return;

    // Add user message
    setMessages((prev) => [...prev, { sender: 'user', text: message }]);

    try {
      // Call backend API
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      const data = await response.json();
      console.log("Bot response:", data);

      // Add bot reply (fix path to answer)
      setMessages((prev) => [...prev, { sender: 'bot', text: data.data.answer }]);
    } catch (error) {
      console.error("Error fetching bot response:", error);
      setMessages((prev) => [...prev, { sender: 'bot', text: "Error: Could not connect to backend." }]);
    }
  };

  return (
    <div>
      {/* <Navbar></Navbar> */}
      <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-r from-indigo-200 via-purple-200 to-pink-200 p-6">
        <div className="w-full max-w-lg bg-white rounded-2xl shadow-xl flex flex-col p-4">
          <h1 className="text-2xl font-bold text-center mb-4 text-indigo-600">CampusGPT</h1>
          {/* Show conversation */}
          <OldChat messages={messages} />

          <ChatBox onSend={handleSend} />
        </div>
      </div>
    </div>

  );
}

export default App;
