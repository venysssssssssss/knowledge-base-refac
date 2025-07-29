import { NextResponse } from 'next/server'
import { cookies } from 'next/headers'

export const maxDuration = 300 // 5 minutos para processamento
export const dynamic = 'force-dynamic'

export async function POST(request: Request) {
    try {
        // Verificar autenticação usando o mesmo cookie da login
        const cookieStore = cookies()
        const sessionToken = (await cookieStore).get('session_token')

        if (!sessionToken || sessionToken.value !== 'token-valido') {
            return NextResponse.json(
                { success: false, message: 'Não autorizado' },
                { status: 401 }
            )
        }

        // Obter o arquivo do FormData
        const formData = await request.formData()
        const file = formData.get('file') as File | null

        if (!file) {
            return NextResponse.json(
                { success: false, message: 'Nenhum arquivo enviado' },
                { status: 400 }
            )
        }

        // Validar tamanho do arquivo (10MB máximo)
        if (file.size > 10 * 1024 * 1024) {
            return NextResponse.json(
                { success: false, message: 'Arquivo muito grande (máximo 10MB)' },
                { status: 400 }
            )
        }

        // Simular processamento do arquivo (1-3 segundos)
        const processingTime = 1000 + Math.random() * 2000
        await new Promise(resolve => setTimeout(resolve, processingTime))

        // Gerar resposta simulada
        const fileType = file.type.includes('pdf') ? 'PDF' :
            file.type.includes('image') ? 'imagem' : 'documento'

        const analysisResult = {
            success: true,
            message: `Análise do ${fileType} concluída. ` +
                `Arquivo "${file.name}" (${formatFileSize(file.size)}) processado com sucesso. ` +
                `Foram identificadas ${Math.floor(Math.random() * 10) + 1} informações relevantes.`,
            metadata: {
                filename: file.name,
                size: file.size,
                type: file.type,
                processedAt: new Date().toISOString()
            }
        }

        return NextResponse.json(analysisResult, { status: 200 })

    } catch (error) {
        console.error('Erro no upload:', error)
        return NextResponse.json(
            { success: false, message: 'Erro no processamento do arquivo' },
            { status: 500 }
        )
    }
}

// Função auxiliar para formatar tamanho do arquivo (mesmo formato usado no frontend)
function formatFileSize(bytes: number) {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}