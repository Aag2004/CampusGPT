export default function OldChat({ messages }) {
  return (
    <div className="flex-1 w-full h-96 overflow-y-auto bg-gray-50 rounded-xl shadow-inner p-4 flex flex-col gap-3 mb-4">
      {messages.map((msg, idx) => (
        <div
          key={idx}
          className={`px-4 py-2 rounded-xl max-w-[75%] text-sm leading-relaxed shadow 
            ${msg.sender === 'user' 
              ? 'bg-indigo-500 text-white self-end rounded-br-none' 
              : 'bg-gray-200 text-gray-800 self-start rounded-bl-none'
            }`}
        >
          {msg.text}
        </div>
      ))}
    </div>
  );
}

