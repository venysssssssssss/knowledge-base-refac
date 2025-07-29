'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';

type Message = {
    id: string;
    text: string;
    sender: 'user' | 'bot';
    timestamp: Date;
    attachments?: Attachment[];
    isProcessing?: boolean;
};

type Attachment = {
    id: string;
    name: string;
    type: 'pdf' | 'image' | 'document';
    size: number;
    progress?: number;
};

export default function ChatPage() {
    const router = useRouter();
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [isClient, setIsClient] = useState(false);

    useEffect(() => {
        setIsClient(true);
        setMessages([
            {
                id: '1',
                text: 'Olá! Como posso te ajudar hoje? Você pode me enviar documentos ou perguntas diretamente.',
                sender: 'bot',
                timestamp: new Date(),
            },
        ]);
    }, []);

    useEffect(() => {
        if (!isClient) return;

        const isAuthenticated = localStorage.getItem('authenticated') === 'true';
        if (!isAuthenticated) {
            router.push('/login');
        } else {
            inputRef.current?.focus();
        }
    }, [router, isClient]);

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const handleSendMessage = async () => {
        if (inputValue.trim() === '') return;

        const userMessage: Message = {
            id: Date.now().toString(),
            text: inputValue,
            sender: 'user',
            timestamp: new Date(),
        };
        setMessages((prev) => [...prev, userMessage]);
        setInputValue('');
        setIsTyping(true);

        // Simular resposta após 1-3 segundos
        setTimeout(() => {
            const botMessage: Message = {
                id: (Date.now() + 1).toString(),
                text: getRandomResponse(),
                sender: 'bot',
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, botMessage]);
            setIsTyping(false);
        }, 1000 + Math.random() * 2000);
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        const file = files[0];
        if (file.size > 10 * 1024 * 1024) {
            alert('O arquivo é muito grande. Tamanho máximo: 10MB');
            return;
        }

        setIsUploading(true);
        setUploadProgress(0);

        // Criar mensagem de upload
        const attachment: Attachment = {
            id: Date.now().toString(),
            name: file.name,
            type: file.type.includes('pdf') ? 'pdf' : file.type.includes('image') ? 'image' : 'document',
            size: file.size,
            progress: 0,
        };

        const userMessage: Message = {
            id: Date.now().toString(),
            text: '',
            sender: 'user',
            timestamp: new Date(),
            attachments: [attachment],
            isProcessing: true
        };

        setMessages((prev) => [...prev, userMessage]);

        // Simular upload
        const uploadInterval = setInterval(() => {
            setUploadProgress((prev) => {
                const newProgress = prev + Math.random() * 10;
                if (newProgress >= 100) {
                    clearInterval(uploadInterval);
                    return 100;
                }
                return newProgress;
            });
        }, 200);

        try {
            // Enviar para a API
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (data.success) {
                // Atualizar mensagem com a resposta
                setMessages(prev => prev.map(msg =>
                    msg.id === userMessage.id
                        ? {
                            ...msg,
                            isProcessing: false,
                            text: data.message
                        }
                        : msg
                ));
            } else {
                throw new Error(data.message || 'Erro no upload');
            }
        } catch (error) {
            setMessages(prev => [...prev, {
                id: Date.now().toString(),
                text: `Erro ao processar arquivo: ${error instanceof Error ? error.message : 'Erro desconhecido'}`,
                sender: 'bot',
                timestamp: new Date(),
            }]);
        } finally {
            setIsUploading(false);
            setUploadProgress(0);
        }
    };

    const getRandomResponse = () => {
        const responses = [
            'Estou analisando sua solicitação. Por favor, aguarde...',
            'Ótimo! Já estou processando as informações...',
            'Recebi seu documento. Vou extrair os dados relevantes...',
            'Ótima pergunta! Deixe-me verificar isso para você...',
            'Estou consultando a base de conhecimento para te ajudar...',
        ];
        return responses[Math.floor(Math.random() * responses.length)];
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    const formatTime = (date: Date) => {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    const formatFileSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    const triggerFileInput = () => {
        fileInputRef.current?.click();
    };

    if (!isClient) {
        return null;
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white">
            {/* Floating background elements with improved styling */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-1/4 left-1/4 w-64 h-64 rounded-full bg-emerald-900/20 blur-3xl animate-float-slow"></div>
                <div className="absolute bottom-1/3 right-1/3 w-96 h-96 rounded-full bg-blue-900/20 blur-3xl animate-float"></div>
                <div className="absolute top-1/3 right-1/4 w-48 h-48 rounded-full bg-purple-900/15 blur-2xl animate-float-reverse"></div>
            </div>

            <div className="relative max-w-6xl mx-auto h-screen flex flex-col">
                {/* Header with improved border */}
                <header className="relative z-10 py-4 px-6 bg-gray-800/70 backdrop-blur-lg rounded-t-xl border-b border-gray-700/70 flex items-center justify-between shadow-lg border-t border-l border-r border-gray-600/30">
                    <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-emerald-500 to-emerald-700 flex items-center justify-center shadow-md">
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                className="h-6 w-6 text-white"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                                />
                            </svg>
                        </div>
                        <div>
                            <h1 className="text-xl font-semibold bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-cyan-400">
                                Assistente de Conhecimento
                            </h1>
                            <p className="text-xs text-gray-400 flex items-center">
                                <span className={`w-2 h-2 rounded-full mr-1 ${isTyping ? 'bg-yellow-400 animate-pulse' : 'bg-emerald-400'}`}></span>
                                {isTyping ? 'Digitando...' : 'Online'}
                            </p>
                        </div>
                    </div>
                    <button className="text-gray-400 hover:text-white transition-transform hover:scale-110">
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-6 w-6"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z"
                            />
                        </svg>
                    </button>
                </header>

                {/* Chat container with improved borders */}
                <div className="relative z-10 flex-1 overflow-hidden bg-gray-800/40 backdrop-blur-lg flex flex-col shadow-lg border-l border-r border-gray-700/50">
                    {/* Messages area with subtle border effect */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4">
                        {messages.map((message) => (
                            <div
                                key={message.id}
                                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div
                                    className={`max-w-xs md:max-w-md lg:max-w-lg xl:max-w-xl rounded-2xl p-4 transition-all duration-300 relative ${message.sender === 'user'
                                        ? 'bg-gradient-to-br from-emerald-600 to-emerald-700 rounded-tr-none shadow-md border border-emerald-500/30'
                                        : 'bg-gray-700/80 rounded-tl-none shadow-md border border-gray-600/30'}`}
                                >
                                    {/* Processing indicator for file messages */}
                                    {message.isProcessing && (
                                        <div className="absolute -top-2 -right-2 bg-yellow-500 text-white text-xs px-2 py-1 rounded-full animate-pulse">
                                            Processando...
                                        </div>
                                    )}

                                    {message.text && <p className="text-white">{message.text}</p>}

                                    {message.attachments?.map((attachment) => (
                                        <div key={attachment.id} className="mt-2 bg-black/20 rounded-lg p-3 border border-gray-700/50">
                                            <div className="flex items-center space-x-3">
                                                <div className={`p-2 rounded-lg ${attachment.type === 'pdf'
                                                    ? 'bg-red-500/20 border border-red-500/30'
                                                    : attachment.type === 'image'
                                                        ? 'bg-blue-500/20 border border-blue-500/30'
                                                        : 'bg-gray-500/20 border border-gray-500/30'}`}>
                                                    <svg
                                                        xmlns="http://www.w3.org/2000/svg"
                                                        className="h-5 w-5"
                                                        fill="none"
                                                        viewBox="0 0 24 24"
                                                        stroke="currentColor"
                                                    >
                                                        {attachment.type === 'pdf' ? (
                                                            <path
                                                                strokeLinecap="round"
                                                                strokeLinejoin="round"
                                                                strokeWidth={2}
                                                                d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                                                            />
                                                        ) : attachment.type === 'image' ? (
                                                            <path
                                                                strokeLinecap="round"
                                                                strokeLinejoin="round"
                                                                strokeWidth={2}
                                                                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                                                            />
                                                        ) : (
                                                            <path
                                                                strokeLinecap="round"
                                                                strokeLinejoin="round"
                                                                strokeWidth={2}
                                                                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                                            />
                                                        )}
                                                    </svg>
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-sm font-medium text-white truncate">{attachment.name}</p>
                                                    <p className="text-xs text-gray-300">{formatFileSize(attachment.size)}</p>
                                                    {message.isProcessing && (
                                                        <div className="w-full bg-gray-700 rounded-full h-1.5 mt-1 overflow-hidden">
                                                            <div
                                                                className="bg-emerald-500 h-1.5 rounded-full transition-all duration-300"
                                                                style={{ width: `${uploadProgress}%` }}
                                                            ></div>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    ))}

                                    <p
                                        className={`text-xs mt-2 ${message.sender === 'user'
                                            ? 'text-emerald-200'
                                            : 'text-gray-400'}`}
                                    >
                                        {formatTime(message.timestamp)}
                                    </p>
                                </div>
                            </div>
                        ))}

                        {isTyping && (
                            <div className="flex justify-start">
                                <div className="bg-gray-700/80 rounded-2xl rounded-tl-none p-4 max-w-xs shadow-md border border-gray-600/30">
                                    <div className="flex space-x-2">
                                        <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"></div>
                                        <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                                        <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                                    </div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input area with improved border */}
                    <div className="p-4 border-t border-gray-700/50 bg-gray-800/40 backdrop-blur-sm">
                        <div className="relative flex items-center">
                            <button
                                onClick={triggerFileInput}
                                className="p-2 text-gray-400 hover:text-emerald-400 transition-colors mr-2 hover:bg-gray-700/50 rounded-lg"
                                title="Anexar arquivo"
                            >
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    className="h-6 w-6"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
                                    />
                                </svg>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    onChange={handleFileUpload}
                                    accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png"
                                    className="hidden"
                                />
                            </button>

                            <input
                                ref={inputRef}
                                type="text"
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="Digite sua mensagem ou envie um arquivo..."
                                className="flex-1 bg-gray-700/50 border border-gray-600 rounded-xl py-3 px-4 pr-12 text-white focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent backdrop-blur-sm transition-all duration-200 placeholder-gray-400"
                            />

                            <button
                                onClick={handleSendMessage}
                                disabled={inputValue.trim() === ''}
                                className={`absolute right-3 p-1 rounded-full transition-all ${inputValue.trim() === ''
                                    ? 'text-gray-500'
                                    : 'text-emerald-400 hover:text-white bg-emerald-600/30 hover:bg-emerald-600/50'}`}
                            >
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    className="h-6 w-6"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                                    />
                                </svg>
                            </button>
                        </div>

                        <p className="text-xs text-gray-500 mt-2 text-center">
                            {isUploading ? (
                                <span className="text-emerald-400 flex items-center justify-center">
                                    <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        className="h-3 w-3 mr-1 animate-spin"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        stroke="currentColor"
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                                        />
                                    </svg>
                                    Enviando arquivo... {Math.round(uploadProgress)}%
                                </span>
                            ) : (
                                'O assistente pode cometer erros. Verifique informações importantes.'
                            )}
                        </p>
                    </div>
                </div>
            </div>

            {/* Adicionando animações CSS */}
            <style jsx global>{`
                @keyframes float {
                    0%, 100% { transform: translateY(0) rotate(0deg); }
                    50% { transform: translateY(-20px) rotate(2deg); }
                }
                @keyframes float-slow {
                    0%, 100% { transform: translateY(0) rotate(0deg); }
                    50% { transform: translateY(-10px) rotate(-1deg); }
                }
                @keyframes float-reverse {
                    0%, 100% { transform: translateY(0) rotate(0deg); }
                    50% { transform: translateY(15px) rotate(-2deg); }
                }
                .animate-float { animation: float 8s ease-in-out infinite; }
                .animate-float-slow { animation: float-slow 12s ease-in-out infinite; }
                .animate-float-reverse { animation: float-reverse 10s ease-in-out infinite; }
            `}</style>
        </div>
    );
}