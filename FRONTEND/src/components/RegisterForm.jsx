import './RegisterForm.css';
import { Link } from 'react-router-dom';

const RegisterForm = () => {
  return (
    <div className="login">
      <div className="hader">
        <span>Create an account</span>
      </div>
      
      <form className='register'>
        <label>Name</label>
        <input 
          type="text" 
          placeholder="이름을 입력하세요." 
          required 
        />
        <label>Email</label>
        <input 
          type="email"
          placeholder="이메일을 입력하세요."
          required
        />
        <label>Password</label>
        <input 
          type="password"
          placeholder="비밀번호를 입력하세요."
          required
        />
        <label>Confirm Password</label>
        <input 
          type="password" 
          placeholder="비밀번호를 재입력하세요." 
          required 
        />
        <input 
          type="button" 
          value="회원가입" 
        />
        
        <span> 
          계정이 있으신가요?
          <Link to="/login"> 로그인</Link>
        </span>
      </form>
    </div>
  );
};

export default RegisterForm;