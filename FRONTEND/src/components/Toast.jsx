import { useEffect } from "react";
import styled, { keyframes } from "styled-components";

const slideAndFade = keyframes`
  0% {
    transform: translate(-50%, 120%);
    opacity: 0;
  }
  10% {
    transform: translate(-50%, 0);
    opacity: 1;
  }
  85% {
    transform: translate(-50%, 0);
    opacity: 1;
  }
  100% {
    transform: translate(-50%, 0);
    opacity: 0;
  }
`;

const Toast = ({ message = "toast", clearToast }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      clearToast?.();
    }, 2000); // ✅ 2초 뒤 토스트 제거

    return () => clearTimeout(timer);
  }, [clearToast]);

  return (
    <StyledWrapper>
      <div className="error">
        <div className="error__icon">!</div>
        <div className="error__title">{message}</div>
      </div>
    </StyledWrapper>
  );
};

const StyledWrapper = styled.div`
  position: fixed;
  bottom: 40px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 2147483647;
  pointer-events: none;
  animation: ${slideAndFade} 2s ease-in-out forwards;

  .error {
    min-width: 280px;
    max-width: 90vw;
    padding: 12px 16px;
    display: flex;
    gap: 10px;
    align-items: center;
    background: #ef665b;
    color: white;
    border-radius: 8px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    font-weight: 600;
    font-size: 14px;
    justify-content: center;
  }

  .error__icon {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.2);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
  }
`;

export default Toast;
