import { createContext, useContext, useState, useCallback } from "react";
import Toast from "../components/Toast"; // 아까 만든 Toast.jsx 컴포넌트 import

const ToastContext = createContext();

export const ToastProvider = ({ children }) => {
  const [toast, setToast] = useState(null);

  // ✅ toast 보여주는 함수
  const showToast = useCallback((message, duration = 3000) => {
    console.log("[ToastProvider] showToast called with:", message);
    setToast(message);
    setTimeout(() => setToast(null), duration);
  }, []);

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      {toast && <Toast message={toast} />}
    </ToastContext.Provider>
  );
};

// ✅ toast 사용 훅
export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return context;
};
