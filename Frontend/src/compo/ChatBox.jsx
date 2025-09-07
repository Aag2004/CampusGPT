import { useState } from 'react';

export default function ChatBox({ onSend }) {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSend(input);
    setInput('');
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="w-full flex gap-2 items-center"
    >
      <input
        type="text"
        placeholder="Type your question..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
        className="flex-1 border border-gray-300 rounded-xl px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400 text-sm"
      />
      <button
        type="submit"
        className="bg-indigo-500 text-white px-5 py-2 rounded-xl hover:bg-indigo-600 transition-all shadow"
      >
        Send
      </button>
    </form>
  );
}
