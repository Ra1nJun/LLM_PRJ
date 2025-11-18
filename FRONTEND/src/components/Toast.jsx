import styled, { keyframes } from "styled-components";

const slideAndFade = keyframes`
  0% { transform: translate(-50%, 120%); opacity: 0; }
  10% { transform: translate(-50%, 0); opacity: 1; }
  85% { transform: translate(-50%, 0); opacity: 1; }
  100% { transform: translate(-50%, 0); opacity: 0; }
`;

const Toast = ({ message, type = "default" }) => {
  return (
    <StyledWrapper type={type}>
      <div className="toast">
        <div className="toast__icon">
            {type === "success" ? "✔" :
            type === "error" ? "✖" :
            type === "warning" ? "⚠" :
            "i"}
        </div>
        <div className="toast__title">{message}</div>
      </div>
    </StyledWrapper>
  );
};

const colors = {
  success: "#2ECC71",
  error: "#E74C3C",
  warning: "#F1C40F",
  info: "#3498DB",
  default: "#555",
};

const StyledWrapper = styled.div`
  position: fixed;
  bottom: 40px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 2147483647;
  pointer-events: none;
  animation: ${slideAndFade} 2s ease-in-out forwards;

  .toast {
    min-width: 280px;
    max-width: 90vw;
    padding: 12px 16px;
    display: flex;
    gap: 10px;
    align-items: center;
    background: ${({ type }) => colors[type] ?? colors.default};
    color: white;
    border-radius: 8px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    font-weight: 600;
    font-size: 14px;
    justify-content: center;
  }

  .toast__icon {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
  }
`;

export default Toast;
