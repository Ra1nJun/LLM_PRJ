import httpClient from './httpClient';

export function chat(message) {
    return httpClient.post('/chat', { message });
}