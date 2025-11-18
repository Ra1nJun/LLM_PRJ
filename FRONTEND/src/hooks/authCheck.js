import { useState, useEffect } from 'react';
import { me } from '../api/authApi';

export function useAuthCheck() {
    const [loggedIn, setLoggedIn] = useState(false);

    useEffect(() => {
        me()
            .then(res => setLoggedIn(res.data.loggedIn))
            .catch(error => {
                console.log(error);
                setLoggedIn(false);
            });
    }, []);

    return loggedIn;
}
