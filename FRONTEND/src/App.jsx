import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ToastProvider } from './context/ToastContext';
import HomePage from './pages/HomePage';
import AboutPage from './pages/AboutPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ChatPage from './pages/ChatPage';
import StaggeredMenu from './components/StaggeredMenu';
import { useAuthCheck } from './hooks/authCheck';
import { logout } from './api/authApi';

function App() {
    const loggedIn = useAuthCheck();
    const handleLogout = async () => {
        try {
            await logout();
            window.location.reload();
        } catch (e) {
            alert('로그아웃 실패: ' + e.message);
        }
    };

    const menuItems = [
        { label: 'Home', ariaLabel: 'Go to home page', link: '/' },
        loggedIn
            ? { label: 'Logout', ariaLabel: 'Logout from your account', onClick: handleLogout }
            : { label: 'Login', ariaLabel: 'Login with your account', link: '/login' },
        { label: 'About', ariaLabel: 'Learn about us', link: '/about' }
    ];

    return (
        <ToastProvider>
            <StaggeredMenu
                    position="right"
                    items={menuItems}
                    displaySocials={true}
                    displayItemNumbering={true}
                    menuButtonColor="#fff"
                    openMenuButtonColor="#fff"
                    changeMenuColorOnOpen={true}
                    colors={['rgb(232, 62, 57)', 'rgb(242, 143, 63)']}
                    accentColor="rgb(242, 143, 63)"
                    onMenuOpen={() => console.log('Menu opened')}
                    onMenuClose={() => console.log('Menu closed')}
                />

            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/about" element={<AboutPage />} />
                <Route path="/login" element={<LoginPage />} />
                <Route path="/register" element={<RegisterPage />} />
                <Route path="/chat" element={<ChatPage />} />
            </Routes>
        </ToastProvider>
    );
}

export default App;