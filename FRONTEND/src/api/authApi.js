import httpClient from './httpClient';

export function login(email, password) {
    return httpClient.post('/auth/login', { email, password });
}

export function logout() {
    return httpClient.post('/auth/logout');
}

export function me() {
    return httpClient.get('/auth/me');
}