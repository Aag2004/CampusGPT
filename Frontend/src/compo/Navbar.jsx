import { useState } from "react";
import { Menu, X, MessageSquare } from "lucide-react";

export default function ChatbotNavbar() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="navbar">
      <div className="navbar-container">
        
        {/* Logo */}
        <div className="navbar-logo">
          <MessageSquare className="logo-icon" />
          <span className="logo-text">CampusGPT</span>
        </div>

        {/* Desktop Menu */}
        <div className="navbar-links">
          <a href="#home">Home</a>
          <a href="#chat">Chat</a>
          <a href="#docs">Docs</a>
          <a href="#about">About</a>
          <button className="login-btn">Login</button>
        </div>

        {/* Mobile Menu Button */}
        <div className="navbar-toggle" onClick={() => setIsOpen(!isOpen)}>
          {isOpen ? <X /> : <Menu />}
        </div>
      </div>

      {/* Mobile Dropdown */}
      {isOpen && (
        <div className="navbar-dropdown">
          <a href="#home">Home</a>
          <a href="#chat">Chat</a>
          <a href="#docs">Docs</a>
          <a href="#about">About</a>
          <button className="login-btn">Login</button>
        </div>
      )}
    </nav>
  );
}
