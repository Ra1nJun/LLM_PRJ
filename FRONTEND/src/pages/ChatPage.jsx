import { useState, useRef, useEffect } from "react";
import { useLocation } from "react-router-dom";
import AnimatedList from "../components/AnimatedList";
import "./ChatPage.css";
import { IoMdSend } from "react-icons/io";

const ChatPage = () => {
  const [messages, setMessages] = useState([]); // 대화 리스트
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);
  const location = useLocation(); // ✅ HomePage에서 전달된 state 받기

  // ✅ HomePage에서 입력된 메시지가 있을 경우 초기 메시지로 추가
  useEffect(() => {
    if (location.state?.userInput) {
      const userMessage = { id: Date.now(), text: location.state.userInput, sender: "user" };
      setMessages([userMessage]);
      sendMessageToBackend(location.state.userInput); // ✅ 처음부터 응답 요청까지 수행
    }
  }, [location.state]);

  // ✅ 새로운 메시지가 추가될 때 자동 스크롤
  useEffect(() => {
    if (chatEndRef.current) {
      setTimeout(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
      }, 100);
    }
  }, [messages]);

  // ✅ 메시지 전송 로직 분리
  const sendMessageToBackend = async (text) => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();

      const botMessage = {
        id: Date.now() + 1,
        text: data.answer || "응답을 불러올 수 없습니다.",
        sender: "bot",
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 2, text: "서버 오류가 발생했습니다.", sender: "bot" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // ✅ 수동 전송 (입력창에서 보낼 때)
  const handleSubmit = async () => {
    const text = inputValue.trim();
    if (!text) return;

    const userMessage = { id: Date.now(), text, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    await sendMessageToBackend(text);
  };

  return (
    <div className="chat-container">
      <div className="chat-window">
        <AnimatedList messages={messages} />
        {loading && (
            <div className="loading">
                <span>응답 생성 중...</span>
                <div className="dot-spinner">
                    <div className="dot-spinner__dot"></div>
                    <div className="dot-spinner__dot"></div>
                    <div className="dot-spinner__dot"></div>
                    <div className="dot-spinner__dot"></div>
                    <div className="dot-spinner__dot"></div>
                    <div className="dot-spinner__dot"></div>
                    <div className="dot-spinner__dot"></div>
                    <div className="dot-spinner__dot"></div>
                </div>
            </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <div className="chat-input-area">
        <textarea
          className="chat-input"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="메시지를 입력하세요..."
          rows={1}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit();
            }
          }}
        />
        <IoMdSend className="send-btn" onClick={handleSubmit} />
      </div>
    </div>
  );
};

export default ChatPage;