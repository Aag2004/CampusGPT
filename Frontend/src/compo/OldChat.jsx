// export default function OldChat({ messages }) {
//   return (
//     <div className="flex-1 w-full h-96 overflow-y-auto bg-gray-50 rounded-xl shadow-inner p-4 flex flex-col gap-3 mb-4">
//       {messages.map((msg, idx) => (
//         <div
//           key={idx}
//           className={`px-4 py-2 rounded-xl max-w-[75%] text-sm leading-relaxed shadow 
//             ${msg.sender === 'user' 
//               ? 'bg-indigo-500 text-white self-end rounded-br-none' 
//               : 'bg-gray-200 text-gray-800 self-start rounded-bl-none'
//             }`}
//         >
//           {msg.text}
//         </div>
//       ))}
//     </div>
//   );
// }

import { useEffect, useRef } from "react";

function OldChat({ messages }) {
  const chatEndRef = useRef(null);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex flex-col space-y-2 mb-4 max-h-96 overflow-y-auto p-2 border rounded-lg bg-gray-50">
      {messages.map((msg, idx) => (
        <div
          key={idx}
          className={`p-2 rounded-xl max-w-xs ${
            msg.sender === "user"
              ? "bg-indigo-500 text-white self-end"
              : "bg-gray-200 text-gray-900 self-start"
          }`}
        >
          {msg.text}
        </div>
      ))}
      {/* Invisible element for auto-scroll */}
      <div ref={chatEndRef} />
    </div>
  );
}

export default OldChat;
