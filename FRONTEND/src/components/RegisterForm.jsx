import './RegisterForm.css';
import { Link } from 'react-router-dom';

const RegisterForm = () => {
  return (
    <div className="login">
      <div className="hader">
        <span>Create an account</span>
      </div>
      
      <form className='register'>
        <label>Full Name</label>
        <input 
          type="text" 
          placeholder="Enter Name" 
          required 
        />
        <label>Email</label>
        <input 
          type="email"
          placeholder="Enter Email"
          required
        />
        <label>Password</label>
        <input 
          type="password"
          placeholder="Enter A Password"
          required
        />
        <label>Confirm Password</label>
        <input 
          type="password" 
          placeholder="Re-Enter Password" 
          required 
        />
        <input 
          type="button" 
          value="Signup" 
        />
        
        <span> 
          Already a member?
          <Link to="/login"> Login Here</Link>
        </span>
      </form>
    </div>
  );
};

export default RegisterForm;