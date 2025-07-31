'use client'

import { useEffect, useRef, useState } from 'react'
import { Card } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'

// Tipo de log unificado
export type LogEntry = {
    timestamp: Date
    type: 'request' | 'response' | 'info' | 'warning' | 'error'
    content: string
    context?: Record<string, any>
}

// Componente visual dos logs - versão melhorada
export function ChatLogger({
    logs,
    maxHeight = '300px',
    floating = false
}: {
    logs: LogEntry[]
    maxHeight?: string
    floating?: boolean
}) {
    const [isOpen, setIsOpen] = useState(!floating)
    const logsEndRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (isOpen && logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
        }
    }, [logs, isOpen])

    const getTypeColor = (type: LogEntry['type']) => {
        switch (type) {
            case 'request': return 'bg-blue-500/20 text-white'
            case 'response': return 'bg-green-500/20 text-white'
            case 'error': return 'bg-red-500/20 text-white'
            case 'warning': return 'bg-yellow-500/20 text-white'
            default: return 'bg-gray-500/20 text-white'
        }
    }

    if (floating) {
        return (
            <div className="fixed bottom-4 right-4 z-50">
                <button
                    onClick={() => setIsOpen(!isOpen)}
                    className={`p-3 rounded-full shadow-lg relative ${logs.some(l => l.type === 'error') ? 'bg-red-900/80 hover:bg-red-800' :
                            'bg-gray-800 hover:bg-gray-700'
                        }`}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    {logs.some(l => l.type === 'error') && (
                        <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse"></span>
                    )}
                </button>

                {isOpen && (
                    <div className="absolute bottom-16 right-0 w-80 h-96 bg-gray-900 rounded-lg shadow-xl flex flex-col border border-gray-700 overflow-hidden">
                        <div className="p-3 border-b border-gray-700 flex justify-between items-center bg-gray-800">
                            <h3 className="font-semibold text-sm">Registros do Chat</h3>
                            <div className="flex items-center space-x-2">
                                <span className={`text-xs px-2 py-1 rounded-full ${logs.some(l => l.type === 'error') ? 'bg-red-500/20 text-red-400' :
                                        logs.some(l => l.type === 'warning') ? 'bg-yellow-500/20 text-yellow-400' :
                                            'bg-green-500/20 text-green-400'
                                    }`}>
                                    {logs.filter(l => l.type === 'error').length} Erros
                                </span>
                                <button onClick={() => setIsOpen(false)} className="text-white hover:text-white">
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                        <div className="flex-1 overflow-y-auto p-3 space-y-2 text-sm">
                            {logs.length === 0 ? (
                                <p className="text-gray-500 text-center mt-10">Nenhum log registrado ainda</p>
                            ) : (
                                logs.map((log, index) => (
                                    <div key={index} className={`p-2 rounded ${getTypeColor(log.type)}`}>
                                        <div className="flex justify-between">
                                            <span className="font-mono text-xs text-white">
                                                {log.timestamp.toLocaleTimeString()}
                                            </span>
                                            <span className={`text-xs font-semibold ${log.type === 'error' ? 'text-red-400' :
                                                    log.type === 'warning' ? 'text-yellow-400' :
                                                        log.type === 'response' ? 'text-green-400' :
                                                            'text-blue-400'
                                                }`}>
                                                {log.type.toUpperCase()}
                                            </span>
                                        </div>
                                        <pre className="mt-1 whitespace-pre-wrap break-words">{log.content}</pre>
                                        {log.context && (
                                            <details className="mt-1 text-white">
                                                <summary className="text-xs cursor-pointer">Contexto</summary>
                                                <pre className="text-xs bg-black/20 p-1 rounded mt-1 overflow-x-auto">
                                                    {JSON.stringify(log.context, null, 2)}
                                                </pre>
                                            </details>
                                        )}
                                    </div>
                                ))
                            )}
                            <div ref={logsEndRef} />
                        </div>
                        <div className="p-2 border-t border-gray-700 text-xs text-gray-500 text-center">
                            {logs.length} eventos registrados
                        </div>
                    </div>
                )}
            </div>
        )
    }

    // Versão não flutuante (inline)
    return (
        <Card className="w-full border-muted p-2">
            <ScrollArea className="w-full pr-2" style={{ maxHeight }}>
                <div className="space-y-2">
                    {logs.map((log, index) => (
                        <div
                            key={index}
                            className={`text-sm rounded-md px-2 py-1 ${getTypeColor(log.type)}`}
                        >
                            <strong className="block text-xs text-muted-foreground mb-1">
                                [{log.timestamp.toLocaleTimeString()}] {log.type.toUpperCase()}
                            </strong>
                            <pre className="whitespace-pre-wrap break-words">{log.content}</pre>
                            {log.context && (
                                <details className="mt-1 text-xs text-muted-foreground">
                                    <summary>Contexto</summary>
                                    <pre className="bg-black/10 p-1 rounded mt-1 overflow-x-auto">
                                        {JSON.stringify(log.context, null, 2)}
                                    </pre>
                                </details>
                            )}
                        </div>
                    ))}
                </div>
            </ScrollArea>
        </Card>
    )
}

// Hook para usar e adicionar logs - versão melhorada
export function useChatLogger(initialLogs: LogEntry[] = []) {
    const [logs, setLogs] = useState<LogEntry[]>(initialLogs)

    const addLog = (type: LogEntry['type'], content: string, context?: Record<string, any>) => {
        const newLog: LogEntry = {
            timestamp: new Date(),
            type,
            content,
            context
        }
        setLogs(prev => [newLog, ...prev].slice(0, 200)) // Limita a 200 logs
    }

    const clearLogs = () => {
        setLogs([])
    }

    return {
        logs,
        addLog,
        clearLogs,
        addRequest: (content: string, context?: Record<string, any>) => addLog('request', content, context),
        addResponse: (content: string, context?: Record<string, any>) => addLog('response', content, context),
        addInfo: (content: string, context?: Record<string, any>) => addLog('info', content, context),
        addWarning: (content: string, context?: Record<string, any>) => addLog('warning', content, context),
        addError: (content: string, context?: Record<string, any>) => addLog('error', content, context)
    }
}