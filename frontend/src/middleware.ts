import { NextResponse, type NextRequest } from 'next/server'

export async function middleware(request: NextRequest) {
    const sessionToken = request.cookies.get('session_token')?.value
    const { pathname } = request.nextUrl

    console.log('Middleware triggered:', {
        path: request.nextUrl.pathname,
        cookies: request.cookies.getAll(),
        headers: Object.fromEntries(request.headers.entries())
    })

    // Rotas públicas
    const publicRoutes = ['/login', '/']
    if (publicRoutes.includes(pathname)) {
        if (sessionToken) {
            return NextResponse.redirect(new URL('/chat', request.url))
        }
        return NextResponse.next()
    }

    // Rotas protegidas
    if (!sessionToken) {
        const loginUrl = new URL('/login', request.url)
        loginUrl.searchParams.set('redirect', pathname)
        return NextResponse.redirect(loginUrl)
    }

    return NextResponse.next()
}

export const config = {
    matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
}