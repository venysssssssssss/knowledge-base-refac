import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';
import { NextRequest } from 'next/server';

const SESSION_COOKIE_NAME = 'session_token';
const TEST_USER = { email: 'tahto@tahto.com', password: 'tahto' };

// Versão otimizada para usar tanto no middleware quanto nos componentes
export async function isAuthenticated(request?: NextRequest): Promise<boolean> {
    if (request) {
        // Para uso no middleware
        return request.cookies.has(SESSION_COOKIE_NAME);
    } else {
        // Para uso em componentes
        const cookieStore = cookies();
        return (await cookieStore).has(SESSION_COOKIE_NAME);
    }
}

export async function login(email: string, password: string) {
    if (email === TEST_USER.email && password === TEST_USER.password) {
        const cookieStore = cookies();
        (await cookieStore).set(SESSION_COOKIE_NAME, 'authenticated', {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            sameSite: 'strict',
            maxAge: 60 * 60 * 24 * 7, // 1 semana
            path: '/',
        });
        return { success: true };
    }
    return { success: false, error: 'Credenciais inválidas' };
}

export async function logout() {
    const cookieStore = cookies();
    (await cookieStore).delete(SESSION_COOKIE_NAME);
    redirect('/login');
}

// Helper para obter o usuário atual (pode ser expandido)
export async function getCurrentUser() {
    const isAuth = await isAuthenticated();
    return isAuth ? { email: TEST_USER.email } : null;
}