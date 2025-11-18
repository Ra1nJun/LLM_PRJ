import httpClient from './httpClient';
import qs from 'qs'; 

export function login(email, password) {
    const data = qs.stringify({ username: email, password });
    return httpClient.post('/auth/login', data, {
    headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    });
}

export function logout() {
    return httpClient.post('/auth/logout');
}

export function me() {
    return httpClient.get('/auth/me');
}