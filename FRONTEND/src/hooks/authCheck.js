import { useState, useEffect } from 'react';
import { me } from '../api/authApi';
import { useLocation } from 'react-router-dom';

export function useAuthCheck() {
    const location = useLocation();
    const initialLoggedIn = location.state && location.state.loggedIn;
    const [loggedIn, setLoggedIn] = useState(initialLoggedIn || false);

    useEffect(() => {
        if (initialLoggedIn) return loggedIn; // 이미 상태를 전달받았으므로 API 검증 생략
        
        me()
            .then(res => setLoggedIn(res.data.loggedIn))
            .catch(error => {
                console.log(error);
                setLoggedIn(false);
            });
    }, [initialLoggedIn]);

    return loggedIn;
}