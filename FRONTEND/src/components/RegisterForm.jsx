import './RegisterForm.css';
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { register } from "../api/userApi";
import { useToast } from '../context/ToastContext';

const RegisterForm = () => {
  const { showToast } = useToast();
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirm_password: "",
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
    
    if (e.target.name === "password" && e.target.value.length > 0 && e.target.value.length < 8) {
      showToast("비밀번호는 최소 8자 이상이어야 합니다.", "warning");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (formData.password !== formData.confirm_password) {
      showToast("비밀번호가 일치하지 않습니다.", "warning");
      return;
    }

    try {
      await register(
        formData.username,
        formData.email,
        formData.password,
        formData.confirm_password
      );

      showToast("회원가입 성공!", "success");
      navigate("/login");
    } catch (error) {
      console.error(error);
      showToast("회원가입 실패", "error");
    }
  };

  return (
    <div className="login">
      <div className="hader">
        <span>Create an account</span>
      </div>

      <form className="register" onSubmit={handleSubmit}>
        <label>Name</label>
        <input
          type="text"
          name="username"
          placeholder="이름을 입력하세요."
          value={formData.username}
          onChange={handleChange}
          required
        />

        <label>Email</label>
        <input
          type="email"
          name="email"
          placeholder="이메일을 입력하세요."
          value={formData.email}
          onChange={handleChange}
          required
        />

        <label>Password</label>
        <input
          type="password"
          name="password"
          placeholder="비밀번호를 입력하세요. (최소 8자)"
          value={formData.password}
          onChange={handleChange}
          required
        />

        <label>Confirm Password</label>
        <input
          type="password"
          name="confirm_password"
          placeholder="비밀번호를 재입력하세요."
          value={formData.confirm_password}
          onChange={handleChange}
          required
        />

        <button type="submit">회원가입</button>

        <span>
          계정이 있으신가요?
          <Link to="/login"> 로그인</Link>
        </span>
      </form>
    </div>
  );
};

export default RegisterForm;
