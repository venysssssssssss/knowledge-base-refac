'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { ChatLogger, useChatLogger } from '@/app/chat/registroschatELK';
import {
    saveMessageToCache,
    loadCachedMessages,
    removeFromCache,
    checkConnectionStatus,
    setupConnectionListener,
    processFileForOffline
} from './chatCache';
import { aiClient, type RAGRequest, type RAGResponse, AIServiceError } from '@/lib/ai-client';
import { AILoadingIndicator, BotTypingIndicator, useLoadingStates } from '@/components/ui/loading';

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
    type: 'pdf' | 'image' | 'document' | 'spreadsheet' | 'presentation';
    size: number;
    progress?: number;
    extension?: string;
};

const ALLOWED_FILE_TYPES = [
    'application/pdf',
    'image/jpeg',
    'image/png',
    'image/gif',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-powerpoint',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'text/plain'
];

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

// Função para gerar IDs únicos usando crypto.randomUUID() ou fallback robusto
const generateUniqueId = () => {
    // Usar crypto.randomUUID() se disponível (mais seguro)
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    
    // Fallback: timestamp + random + incremento + performance
    const timestamp = Date.now();
    const random = Math.random().toString(36).substr(2, 9);
    const performance = typeof window !== 'undefined' && window.performance 
        ? window.performance.now().toString(36).replace('.', '') 
        : '';
    const increment = (Date.now() % 1000000).toString(36);
    
    return `${timestamp}-${random}-${performance}-${increment}`;
};

export default function ChatPage() {
    const router = useRouter();
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [isOnline, setIsOnline] = useState(true);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [isClient, setIsClient] = useState(false);

    // Usando o hook do logger
    const { logs, addLog, addError, addWarning, addInfo, addRequest, addResponse, clearLogs } = useChatLogger();
    
    // Hook para estados de loading
    const { isLoading, withLoading } = useLoadingStates();

    // Função para adicionar mensagem de forma segura, evitando duplicatas
    const addMessage = (newMessage: Message) => {
        setMessages((prev) => {
            // Verificar se a mensagem já existe
            if (prev.some(msg => msg.id === newMessage.id)) {
                console.warn(`Mensagem com ID ${newMessage.id} já existe, ignorando duplicata`);
                return prev;
            }
            return [...prev, newMessage];
        });
    };

    // Função para atualizar mensagem existente
    const updateMessage = (messageId: string, updater: (msg: Message) => Message) => {
        setMessages(prev => prev.map(msg => 
            msg.id === messageId ? updater(msg) : msg
        ));
    };

    useEffect(() => {
        setIsClient(true);

        // Verificar status de conexão inicial
        setIsOnline(checkConnectionStatus());

        // Carregar mensagens do cache
        const loadCachedData = async () => {
            try {
                const cachedMessages = await loadCachedMessages();
                if (cachedMessages.length > 0) {
                    setMessages(cachedMessages);
                    addInfo('Mensagens carregadas do cache offline', { count: cachedMessages.length });
                } else {
                    setMessages([{
                        id: '1',
                        text: 'Olá! Como posso te ajudar hoje? Você pode me enviar documentos ou perguntas diretamente.',
                        sender: 'bot',
                        timestamp: new Date(),
                    }]);
                }
            } catch (error) {
                addError('Erro ao carregar mensagens do cache', {
                    error: error instanceof Error ? error.message : 'Erro desconhecido'
                });
                setMessages([{
                    id: '1',
                    text: 'Olá! Como posso te ajudar hoje? Você pode me enviar documentos ou perguntas diretamente.',
                    sender: 'bot',
                    timestamp: new Date(),
                }]);
            }
        };

        loadCachedData();
        addInfo('Sessão de chat iniciada');

        // Configurar listener de conexão
        const cleanupListener = setupConnectionListener((online) => {
            setIsOnline(online);
            addInfo(online ? 'Conexão restaurada' : 'Modo offline ativado');

            if (online) {
                // Tentar sincronizar mensagens pendentes quando a conexão voltar
                // (Implementar lógica de sincronização conforme necessário)
            }
        });

        return () => {
            cleanupListener();
        };
    }, []);

    useEffect(() => {
        if (!isClient) return;

        const isAuthenticated = localStorage.getItem('authenticated') === 'true';
        if (!isAuthenticated) {
            addWarning('Usuário não autenticado, redirecionando para login');
            router.push('/login');
        } else {
            inputRef.current?.focus();
            addInfo('Usuário autenticado, chat pronto para uso');
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
            id: generateUniqueId(),
            text: inputValue,
            sender: 'user',
            timestamp: new Date(),
        };

        addRequest('Mensagem do usuário', {
            messageId: userMessage.id,
            messageLength: userMessage.text.length,
            truncatedText: userMessage.text.length > 50
                ? userMessage.text.substring(0, 50) + '...'
                : userMessage.text
        });

        // Salvar no cache imediatamente
        await saveMessageToCache(userMessage);
        addMessage(userMessage);
        setInputValue('');
        setIsTyping(true);

        try {
            if (!isOnline) {
                // Modo offline - simular resposta do bot
                await new Promise(resolve => setTimeout(resolve, 1000));

                const botMessage: Message = {
                    id: generateUniqueId(),
                    text: 'Estou offline no momento. Sua mensagem foi salva e será processada quando a conexão for restaurada.',
                    sender: 'bot',
                    timestamp: new Date(),
                };

                await saveMessageToCache(botMessage);
                addMessage(botMessage);
                addWarning('Modo offline - mensagem salva localmente');
                return;
            }

            // NOVA INTEGRAÇÃO COM IA REAL 
            const ragRequest: RAGRequest = {
                question: userMessage.text,
                max_tokens: 512,
                temperature: 0.7,
                search_limit: 3,
                score_threshold: 0.6
            };

            const startTime = Date.now();
            addInfo('Enviando pergunta para IA...', { question: userMessage.text });

            const aiResponse: RAGResponse = await withLoading('ai-query', async () => {
                return await aiClient.askQuestion(ragRequest);
            });
            
            const endTime = Date.now();

            // Verificar se a resposta é válida antes de processar
            if (!aiResponse || !aiResponse.answer) {
                throw new AIServiceError('Resposta inválida recebida do serviço de IA');
            }

            const botMessage: Message = {
                id: generateUniqueId(),
                text: aiResponse.answer,
                sender: 'bot',
                timestamp: new Date(),
            };

            addResponse('Resposta da IA recebida', {
                userMessageId: userMessage.id,
                botMessageId: botMessage.id,
                responseTimeMs: endTime - startTime,
                tokensUsed: aiResponse.tokens_used || 0,
                sourcesFound: aiResponse.sources?.length || 0,
                processingTime: aiResponse.processing_time || 0,
                searchTime: aiResponse.search_time || 0,
                generationTime: aiResponse.generation_time || 0
            });

            // Salvar mensagem no cache e atualizar estado
            try {
                await saveMessageToCache(botMessage);
                addMessage(botMessage);

                // Log das fontes encontradas para debug (sem bloquear o fluxo principal)
                if (aiResponse.sources && aiResponse.sources.length > 0) {
                    addInfo('Fontes utilizadas na resposta', {
                        sources: aiResponse.sources.map(source => ({
                            filename: source.metadata.filename,
                            score: source.score,
                            contentPreview: source.content.substring(0, 100) + '...'
                        }))
                    });
                }
            } catch (cacheError) {
                // Se falhar ao salvar no cache, ainda exibir a mensagem para o usuário
                console.warn('Falha ao salvar no cache, mas mensagem será exibida:', cacheError);
                addMessage(botMessage);
                
                addWarning('Falha ao salvar no cache', {
                    error: cacheError instanceof Error ? cacheError.message : 'Erro desconhecido'
                });
            }

        } catch (error) {
            console.error('Erro durante processamento da mensagem:', error);
            
            let errorMessage = 'Erro inesperado ao processar sua mensagem';
            let shouldDisplayError = true;
            
            if (error instanceof AIServiceError) {
                addError('Erro no serviço de IA', {
                    service: error.service,
                    status: error.status,
                    message: error.message,
                    inputMessage: userMessage.text
                });
                
                if (error.status === 503) {
                    errorMessage = 'Os serviços de IA estão temporariamente indisponíveis. Tente novamente em alguns instantes.';
                } else if (error.status === 429) {
                    errorMessage = 'Muitas requisições. Aguarde um momento antes de tentar novamente.';
                } else if (error.status === 401) {
                    errorMessage = 'Sessão expirada. Faça login novamente.';
                } else if (error.status === 404) {
                    errorMessage = 'Serviço não encontrado. Verifique se os serviços de IA estão rodando.';
                } else {
                    errorMessage = `Erro no serviço de IA: ${error.message}`;
                }
            } else if (error instanceof Error) {
                // Verificar se é um erro conhecido que não deve exibir mensagem de erro
                if (error.message.includes('AbortError') || error.message.includes('cancelled')) {
                    // Operação cancelada pelo usuário, não exibir erro
                    shouldDisplayError = false;
                    addInfo('Operação cancelada pelo usuário');
                } else {
                    addError('Falha ao processar mensagem', {
                        error: error.message,
                        inputMessage: userMessage.text,
                        stack: error.stack
                    });
                    
                    if (error.message.includes('NetworkError') || error.message.includes('fetch')) {
                        errorMessage = 'Erro de conexão. Verifique sua internet e tente novamente.';
                    }
                }
            } else {
                addError('Erro desconhecido', {
                    error: String(error),
                    inputMessage: userMessage.text
                });
            }

            // Só exibir mensagem de erro se necessário
            if (shouldDisplayError) {
                const errorBotMessage: Message = {
                    id: generateUniqueId(),
                    text: errorMessage,
                    sender: 'bot',
                    timestamp: new Date(),
                };

                try {
                    await saveMessageToCache(errorBotMessage);
                } catch (cacheError) {
                    console.warn('Falha ao salvar mensagem de erro no cache:', cacheError);
                }
                
                addMessage(errorBotMessage);
            }
        } finally {
            setIsTyping(false);
        }
    };

    const getFileType = (file: File): Attachment['type'] => {
        if (file.type.includes('pdf')) return 'pdf';
        if (file.type.includes('image')) return 'image';
        if (file.type.includes('spreadsheet') || file.type.includes('excel')) return 'spreadsheet';
        if (file.type.includes('presentation') || file.type.includes('powerpoint')) return 'presentation';
        if (file.type.includes('word') || file.type.includes('document')) return 'document';
        return 'document';
    };

    const getFileExtension = (fileName: string): string => {
        return fileName.split('.').pop()?.toLowerCase() || '';
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        const file = files[0];
        const fileExtension = getFileExtension(file.name);

        addInfo('Upload de arquivo iniciado', {
            fileName: file.name,
            fileType: file.type,
            fileSize: file.size,
            fileExtension
        });

        // Validar tipo de arquivo
        if (!ALLOWED_FILE_TYPES.includes(file.type)) {
            addError('Tipo de arquivo não suportado', {
                fileName: file.name,
                fileType: file.type,
                allowedTypes: ALLOWED_FILE_TYPES
            });

            const errorMessage: Message = {
                id: generateUniqueId(),
                text: `Erro: Tipo de arquivo não suportado (${fileExtension || file.type}). Formatos aceitos: PDF, DOC, DOCX, XLS, XLSX, PPT, PPTX, JPG, PNG, GIF, TXT`,
                sender: 'bot',
                timestamp: new Date(),
            };

            await saveMessageToCache(errorMessage);
            addMessage(errorMessage);
            return;
        }

        // Validar tamanho do arquivo
        if (file.size > MAX_FILE_SIZE) {
            addWarning('Upload rejeitado - arquivo muito grande', {
                fileName: file.name,
                fileSize: file.size,
                maxAllowed: '10MB'
            });

            const errorMessage: Message = {
                id: generateUniqueId(),
                text: `Erro: O arquivo é muito grande (${formatFileSize(file.size)}). Tamanho máximo permitido: 10MB`,
                sender: 'bot',
                timestamp: new Date(),
            };

            await saveMessageToCache(errorMessage);
            addMessage(errorMessage);
            return;
        }

        setIsUploading(true);
        setUploadProgress(0);

        // Processar arquivo para armazenamento offline
        const processedFile = await processFileForOffline(file);

        const attachment: Attachment = {
            id: processedFile.id,
            name: processedFile.name,
            type: processedFile.type,
            size: processedFile.size,
            progress: 0,
            extension: processedFile.extension
        };

        const userMessage: Message = {
            id: generateUniqueId(),
            text: '',
            sender: 'user',
            timestamp: new Date(),
            attachments: [attachment],
            isProcessing: true
        };

        // Salvar no cache imediatamente
        await saveMessageToCache(userMessage);
        addMessage(userMessage);

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
            if (!isOnline) {
                // Modo offline - apenas armazenar localmente
                await new Promise(resolve => setTimeout(resolve, 1500));

                const botMessage: Message = {
                    id: generateUniqueId(),
                    text: 'Arquivo recebido em modo offline. Será processado quando a conexão for restaurada.',
                    sender: 'bot',
                    timestamp: new Date(),
                };

                await saveMessageToCache(botMessage);
                updateMessage(userMessage.id, (msg) => ({
                    ...msg,
                    isProcessing: false,
                    text: 'Arquivo salvo localmente (modo offline)'
                }));
                addMessage(botMessage);
                addWarning('Modo offline - arquivo salvo localmente');
                return;
            }

            // ✨ NOVA INTEGRAÇÃO COM UPLOAD REAL ✨
            addInfo('Enviando arquivo para processamento...', { 
                fileName: file.name, 
                fileSize: file.size 
            });

            const uploadResponse = await withLoading('file-upload', async () => {
                return await aiClient.uploadDocument(file);
            });

            if (uploadResponse.success) {
                addInfo('Upload de arquivo concluído com sucesso', {
                    fileName: file.name,
                    fileSize: file.size,
                    processingTimeMs: Date.now() - userMessage.timestamp.getTime(),
                    documentId: uploadResponse.document_id,
                    chunksCreated: uploadResponse.chunks_created
                });

                const updatedMessage: Message = {
                    ...userMessage,
                    isProcessing: false,
                    text: uploadResponse.message || 'Arquivo processado com sucesso'
                };

                await saveMessageToCache(updatedMessage);
                updateMessage(userMessage.id, () => updatedMessage);

                // Resposta automática do bot sobre o documento processado
                const botMessage: Message = {
                    id: generateUniqueId(),
                    text: `✅ Documento "${file.name}" foi processado e indexado com sucesso! ${uploadResponse.chunks_created ? `Foram criados ${uploadResponse.chunks_created} fragmentos para busca.` : ''} Agora você pode fazer perguntas sobre o conteúdo deste documento.`,
                    sender: 'bot',
                    timestamp: new Date(),
                };

                await saveMessageToCache(botMessage);
                addMessage(botMessage);
            } else {
                throw new Error(uploadResponse.message || 'Erro no processamento do arquivo');
            }
        } catch (error) {
            addError('Falha no upload de arquivo', {
                fileName: file.name,
                error: error instanceof Error ? error.message : 'Erro desconhecido',
                fileType: file.type,
                fileSize: file.size
            });

            let errorMessage = 'Erro ao processar arquivo';
            
            if (error instanceof AIServiceError) {
                if (error.status === 413) {
                    errorMessage = 'Arquivo muito grande. Tamanho máximo: 10MB';
                } else if (error.status === 415) {
                    errorMessage = 'Tipo de arquivo não suportado';
                } else if (error.status === 503) {
                    errorMessage = 'Serviço de processamento indisponível. Tente novamente em alguns instantes.';
                } else {
                    errorMessage = `Erro no upload: ${error.message}`;
                }
            } else if (error instanceof Error) {
                errorMessage = `Erro ao processar arquivo: ${error.message}`;
            }

            const errorBotMessage: Message = {
                id: generateUniqueId(),
                text: errorMessage,
                sender: 'bot',
                timestamp: new Date(),
            };

            await saveMessageToCache(errorBotMessage);
            addMessage(errorBotMessage);
        } finally {
            setIsUploading(false);
            setUploadProgress(0);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
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

    const getFileIcon = (type: Attachment['type']) => {
        switch (type) {
            case 'pdf':
                return (
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                    />
                );
            case 'image':
                return (
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                );
            case 'spreadsheet':
                return (
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                );
            case 'presentation':
                return (
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"
                    />
                );
            default:
                return (
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                );
        }
    };

    const getFileTypeColor = (type: Attachment['type']) => {
        switch (type) {
            case 'pdf': return 'bg-red-500/20 border border-red-500/30';
            case 'image': return 'bg-blue-500/20 border border-blue-500/30';
            case 'spreadsheet': return 'bg-green-500/20 border border-green-500/30';
            case 'presentation': return 'bg-purple-500/20 border border-purple-500/30';
            default: return 'bg-gray-500/20 border border-gray-500/30';
        }
    };

    if (!isClient) {
        return null;
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white">
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-1/4 left-1/4 w-64 h-64 rounded-full bg-emerald-900/20 blur-3xl animate-float-slow"></div>
                <div className="absolute bottom-1/3 right-1/3 w-96 h-96 rounded-full bg-blue-900/20 blur-3xl animate-float"></div>
                <div className="absolute top-1/3 right-1/4 w-48 h-48 rounded-full bg-purple-900/15 blur-2xl animate-float-reverse"></div>
            </div>

            <div className="relative max-w-6xl mx-auto h-screen flex flex-col">
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
                                <span className={`w-2 h-2 rounded-full mr-1 ${isTyping ? 'bg-yellow-400 animate-pulse' : isOnline ? 'bg-emerald-400' : 'bg-gray-400'}`}></span>
                                {isTyping ? 'Digitando...' : isOnline ? 'Online' : 'Offline'}
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

                <div className="relative z-10 flex-1 overflow-hidden bg-gray-800/40 backdrop-blur-lg flex flex-col shadow-lg border-l border-r border-gray-700/50">
                    {!isOnline && (
                        <div className="bg-yellow-500/20 text-yellow-300 text-sm p-2 text-center border-b border-yellow-500/30">
                            Você está offline. As mensagens serão salvas localmente e enviadas quando a conexão for restaurada.
                        </div>
                    )}

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
                                    {message.isProcessing && (
                                        <div className="absolute -top-2 -right-2 bg-yellow-500 text-white text-xs px-2 py-1 rounded-full animate-pulse">
                                            Processando...
                                        </div>
                                    )}

                                    {message.text && <p className="text-white">{message.text}</p>}

                                    {message.attachments?.map((attachment) => (
                                        <div key={attachment.id} className="mt-2 bg-black/20 rounded-lg p-3 border border-gray-700/50">
                                            <div className="flex items-center space-x-3">
                                                <div className={`p-2 rounded-lg ${getFileTypeColor(attachment.type)}`}>
                                                    <svg
                                                        xmlns="http://www.w3.org/2000/svg"
                                                        className="h-5 w-5"
                                                        fill="none"
                                                        viewBox="0 0 24 24"
                                                        stroke="currentColor"
                                                    >
                                                        {getFileIcon(attachment.type)}
                                                    </svg>
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-sm font-medium text-white truncate">{attachment.name}</p>
                                                    <div className="flex justify-between items-center">
                                                        <p className="text-xs text-gray-300">
                                                            {attachment.extension ? attachment.extension.toUpperCase() : ''} • {formatFileSize(attachment.size)}
                                                        </p>
                                                    </div>
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
                            <div className="mb-4">
                                <BotTypingIndicator />
                            </div>
                        )}

                        {isLoading('ai-query') && (
                            <div className="mb-4">
                                <AILoadingIndicator 
                                    type="thinking" 
                                    message="Analisando sua pergunta..."
                                />
                            </div>
                        )}

                        {isLoading('file-upload') && (
                            <div className="mb-4">
                                <AILoadingIndicator 
                                    type="uploading" 
                                    message="Enviando seu documento..."
                                    progress={uploadProgress}
                                />
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    <div className="p-4 border-t border-gray-700/50 bg-gray-800/40 backdrop-blur-sm">
                        <div className="relative flex items-center">
                            <button
                                onClick={triggerFileInput}
                                className="p-2 text-gray-400 hover:text-emerald-400 transition-colors mr-2 hover:bg-gray-700/50 rounded-lg"
                                title="Anexar arquivo"
                                disabled={isUploading}
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
                                    accept=".pdf,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.jpg,.jpeg,.png,.gif,.txt"
                                    className="hidden"
                                    disabled={isUploading}
                                />
                            </button>

                            <input
                                ref={inputRef}
                                type="text"
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder={isOnline
                                    ? "Digite sua mensagem ou envie um arquivo..."
                                    : "Modo offline - mensagens serão salvas localmente"}
                                className="flex-1 bg-gray-700/50 border border-gray-600 rounded-xl py-3 px-4 pr-12 text-white focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent backdrop-blur-sm transition-all duration-200 placeholder-gray-400"
                                disabled={isUploading}
                            />

                            <button
                                onClick={handleSendMessage}
                                disabled={inputValue.trim() === '' || isUploading}
                                className={`absolute right-3 p-1 rounded-full transition-all ${inputValue.trim() === '' || isUploading
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
                            ) : !isOnline ? (
                                <span className="text-yellow-400">Modo offline ativado</span>
                            ) : (
                                'O assistente pode cometer erros. Verifique informações importantes.'
                            )}
                        </p>
                    </div>
                </div>
            </div>

            <ChatLogger logs={logs} floating />

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