import { NextResponse } from 'next/server'
import { cookies } from 'next/headers'

export async function POST(request: Request) {
    try {
        const { email, password } = await request.json()

        // Verificação de credenciais
        if (email !== 'tahto@tahto.com' || password !== 'tahto') {
            return NextResponse.json(
                { success: false, message: 'Credenciais inválidas' },
                { status: 401 }
            )
        }

        // Configuração do cookie
        (await cookies()).set('session_token', 'token-valido', {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            sameSite: 'lax',
            path: '/',
            maxAge: 60 * 60 * 24 * 7, // 1 semana
        })

        return NextResponse.json(
            { success: true, redirectUrl: '/chat' },
            { status: 200 }
        )

    } catch (error) {
        return NextResponse.json(
            { success: false, message: 'Erro no servidor' },
            { status: 500 }
        )
    }
}