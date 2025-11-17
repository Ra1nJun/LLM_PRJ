import httpClient from './httpClient';

export function register(username, email, password, confirm_password) {
    return httpClient.post('/users', { username, email, password, confirm_password });
}