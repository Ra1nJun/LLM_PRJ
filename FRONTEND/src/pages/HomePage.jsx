import { useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useToast } from '../context/ToastContext';
import CurvedLoop from '../components/CurvedLoop';
import './HomePage.css'
import { IoMdSend } from "react-icons/io";
import logo from "../assets/dog_256.png"

const HomePage = () => {
    const textareaRef = useRef(null);
    const navigate = useNavigate();
    const { showToast } = useToast();

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = '40px';
        }
    }, []);

    const handleInput = () => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = '0px'; // 완전히 초기화
            const scrollHeight = textarea.scrollHeight;
            const computedStyle = window.getComputedStyle(textarea);
            const paddingTop = parseFloat(computedStyle.paddingTop);
            const paddingBottom = parseFloat(computedStyle.paddingBottom);
            textarea.style.height = `${scrollHeight - paddingTop - paddingBottom}px`;
        }
    };

    const handleSubmit = () => {
        const text = textareaRef.current?.value.trim();
        if (!text) {
            showToast("입력값이 비어 있습니다.");
            return;
        }

        navigate('/chat', { state: { userInput: text } });
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();  // 기본 줄바꿈 동작 방지
            handleSubmit();
        }
    };

    return (
        <div className="homepage-bg">
            <CurvedLoop 
                marqueeText="Haru Dang ✦ Dog Care AI ✦"
                speed={1}
                curveAmount={400}
                interactive={false}
                direction="left"
            />
            <div className="page-center">
                <img src={logo} alt="강아지 로고" className="logo-style" />
                <div className="input-guide">
                    반려견에 대해 궁금한 것이 있으신가요?
                </div>
                <div className="input-container">
                    <textarea
                        ref={textareaRef}
                        placeholder="반려견의 성장 & 질병에 관해 물어보세요! (최대 700자)"
                        className="queryinput"
                        name="text"
                        rows={1}
                        onInput={handleInput}
                        onKeyPress={handleKeyPress}
                        maxLength={700}
                    />
                    <IoMdSend 
                        className="send-icon"
                        onClick={handleSubmit}
                    />
                </div>
            </div>
        </div>
    );
};

export default HomePage;