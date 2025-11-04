import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import AboutPage from './pages/AboutPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import StaggeredMenu from './components/StaggeredMenu';


function App() {
    const menuItems = [
        { label: 'Home', ariaLabel: 'Go to home page', link: '/' },
        { label: 'Login', ariaLabel: 'Login with your account', link: '/login' },
        { label: 'About', ariaLabel: 'Learn about us', link: '/about' }
    ];

    return (
        <Router>
            <StaggeredMenu
                    position="right"
                    items={menuItems}
                    displaySocials={true}
                    displayItemNumbering={true}
                    menuButtonColor="#fff"
                    openMenuButtonColor="#fff"
                    changeMenuColorOnOpen={true}
                    colors={['rgb(253, 198, 11)', 'rgb(232, 62, 57)']}
                    accentColor="rgb(242, 143, 63)"
                    onMenuOpen={() => console.log('Menu opened')}
                    onMenuClose={() => console.log('Menu closed')}
                />

            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/about" element={<AboutPage />} />
                <Route path="/login" element={<LoginPage />} />
                <Route path="/register" element={<RegisterPage />} />
            </Routes>
        </Router>

    );
}

export default App;