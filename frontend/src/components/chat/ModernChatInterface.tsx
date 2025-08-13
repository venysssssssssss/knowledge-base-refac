'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence, Variants } from 'framer-motion';
import { 
  FiSend, 
  FiPaperclip, 
  FiMoreVertical, 
  FiTrash2, 
  FiRefreshCw,
  FiUser,
  FiWifi,
  FiWifiOff,
  FiClock,
  FiCheck,
  FiAlertCircle,
  FiFile,
  FiImage,
  FiFileText,
  FiBarChart,
  FiDownload,
  FiEye,
  FiMessageSquare,
  FiZap,
  FiShield,
  FiCpu,
  FiActivity
} from 'react-icons/fi';
import { HiOutlineDesktopComputer } from 'react-icons/hi';
import { RiRobotLine, RiCheckDoubleLine } from 'react-icons/ri';
import { ScrollArea } from '@/components/ui/scroll-area';

// Interfaces
interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: Date;
  status: 'sending' | 'sent' | 'delivered' | 'error';
  attachments?: Attachment[];
  metadata?: {
    sources?: Source[];
    tokens_used?: number;
    processing_time?: number;
    search_time?: number;
    generation_time?: number;
  };
}

interface Attachment {
  id: string;
  name: string;
  type: 'pdf' | 'image' | 'document' | 'spreadsheet' | 'presentation' | 'text';
  size: number;
  url?: string;
  progress?: number;
}

interface Source {
  content: string;
  score: number;
  metadata: {
    filename: string;
    page?: number;
    chunk_index?: number;
  };
}

interface ChatStats {
  totalMessages: number;
  avgResponseTime: number;
  tokensUsed: number;
  documentsProcessed: number;
}

// Animation variants
const messageVariants: Variants = {
  hidden: { 
    opacity: 0, 
    y: 20,
    scale: 0.95
  },
  visible: { 
    opacity: 1, 
    y: 0,
    scale: 1,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 30
    }
  },
  exit: {
    opacity: 0,
    y: -10,
    scale: 0.95,
    transition: {
      duration: 0.2
    }
  }
};

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const typingVariants: Variants = {
  hidden: { opacity: 0, y: 10 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 400,
      damping: 10
    }
  }
};

// Utility functions
const formatTime = (date: Date): string => {
  return new Intl.DateTimeFormat('pt-BR', {
    hour: '2-digit',
    minute: '2-digit'
  }).format(date);
};

const formatFileSize = (bytes: number): string => {
  const sizes = ['B', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 B';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${Math.round(bytes / Math.pow(1024, i) * 100) / 100} ${sizes[i]}`;
};

const getFileIcon = (type: Attachment['type']) => {
  const icons = {
    pdf: FiFile,
    image: FiImage,
    document: FiFileText,
    spreadsheet: FiBarChart,
    presentation: HiOutlineDesktopComputer,
    text: FiFileText
  };
  return icons[type] || FiFile;
};

const getFileTypeColor = (type: Attachment['type']): string => {
  const colors = {
    pdf: 'from-red-500/20 to-red-600/20 border-red-500/30',
    image: 'from-blue-500/20 to-blue-600/20 border-blue-500/30',
    document: 'from-green-500/20 to-green-600/20 border-green-500/30',
    spreadsheet: 'from-emerald-500/20 to-emerald-600/20 border-emerald-500/30',
    presentation: 'from-purple-500/20 to-purple-600/20 border-purple-500/30',
    text: 'from-gray-500/20 to-gray-600/20 border-gray-500/30'
  };
  return colors[type] || colors.text;
};

// Components
const TypingIndicator = () => (
  <motion.div
    variants={typingVariants}
    initial="hidden"
    animate="visible"
    exit="hidden"
    className="flex items-center space-x-2 text-emerald-400"
  >
    <div className="flex space-x-1">
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="w-2 h-2 bg-emerald-400 rounded-full"
          animate={{
            scale: [1, 1.5, 1],
            opacity: [0.5, 1, 0.5]
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            delay: i * 0.2
          }}
        />
      ))}
    </div>
    <span className="text-sm">AI está digitando...</span>
  </motion.div>
);

const MessageStatus = ({ status }: { status: Message['status'] }) => {
  const statusConfig = {
    sending: { icon: FiClock, color: 'text-gray-400', tooltip: 'Enviando...' },
    sent: { icon: FiCheck, color: 'text-gray-400', tooltip: 'Enviado' },
    delivered: { icon: RiCheckDoubleLine, color: 'text-emerald-400', tooltip: 'Entregue' },
    error: { icon: FiAlertCircle, color: 'text-red-400', tooltip: 'Erro ao enviar' }
  };

  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <div className={`${config.color}`} title={config.tooltip}>
      <Icon size={12} />
    </div>
  );
};

const AttachmentCard = ({ attachment }: { attachment: Attachment }) => {
  const Icon = getFileIcon(attachment.type);
  const colorClass = getFileTypeColor(attachment.type);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`relative bg-gradient-to-br ${colorClass} rounded-xl p-4 border backdrop-blur-sm`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3 flex-1 min-w-0">
          <div className="flex-shrink-0">
            <Icon className="w-8 h-8 text-white" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-white truncate">
              {attachment.name}
            </p>
            <p className="text-xs text-white/70">
              {attachment.type.toUpperCase()} • {formatFileSize(attachment.size)}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2 flex-shrink-0">
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
            title="Visualizar"
          >
            <FiEye className="w-4 h-4 text-white" />
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
            title="Download"
          >
            <FiDownload className="w-4 h-4 text-white" />
          </motion.button>
        </div>
      </div>
      
      {attachment.progress !== undefined && attachment.progress < 100 && (
        <div className="mt-3">
          <div className="flex justify-between text-xs text-white/70 mb-1">
            <span>Enviando...</span>
            <span>{Math.round(attachment.progress)}%</span>
          </div>
          <div className="w-full bg-black/20 rounded-full h-1.5 overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-emerald-400 to-cyan-400 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${attachment.progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>
      )}
    </motion.div>
  );
};

const MessageBubble = ({ message }: { message: Message }) => {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  if (isSystem) {
    return (
      <motion.div
        variants={messageVariants}
        initial="hidden"
        animate="visible"
        exit="exit"
        className="flex justify-center my-4"
      >
        <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-xl px-4 py-2 max-w-md">
          <div className="flex items-center space-x-2">
            <FiAlertCircle className="w-4 h-4 text-yellow-400 flex-shrink-0" />
            <p className="text-sm text-yellow-300">{message.content}</p>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      variants={messageVariants}
      initial="hidden"
      animate="visible"
      exit="exit"
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
    >
      <div className="flex items-end space-x-3 max-w-[80%]">
        {!isUser && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center shadow-lg"
          >
            <RiRobotLine className="w-5 h-5 text-white" />
          </motion.div>
        )}
        
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
          <motion.div
            whileHover={{ scale: 1.02 }}
            className={`relative rounded-2xl px-4 py-3 max-w-lg shadow-lg ${
              isUser
                ? 'bg-gradient-to-br from-emerald-500 to-emerald-600 text-white rounded-br-md'
                : 'bg-gradient-to-br from-gray-700 to-gray-800 text-white rounded-bl-md border border-gray-600/50'
            }`}
          >
            {/* Glass morphism effect */}
            <div className="absolute inset-0 bg-white/5 rounded-2xl backdrop-blur-sm" />
            
            <div className="relative z-10">
              {message.content && (
                <p className="text-sm leading-relaxed whitespace-pre-wrap">
                  {message.content}
                </p>
              )}
              
              {message.attachments && message.attachments.length > 0 && (
                <div className="mt-3 space-y-2">
                  {message.attachments.map((attachment) => (
                    <AttachmentCard key={attachment.id} attachment={attachment} />
                  ))}
                </div>
              )}
              
              {message.metadata?.sources && message.metadata.sources.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="mt-3 pt-3 border-t border-white/10"
                >
                  <p className="text-xs text-white/60 mb-2 flex items-center">
                    <FiShield className="w-3 h-3 mr-1" />
                    Fontes consultadas ({message.metadata.sources.length})
                  </p>
                  <div className="space-y-1">
                    {message.metadata.sources.slice(0, 3).map((source, index) => (
                      <div
                        key={index}
                        className="text-xs bg-black/20 rounded-lg p-2 border border-white/10"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-white/80">
                            {source.metadata.filename}
                          </span>
                          <span className="text-emerald-300">
                            {Math.round(source.score * 100)}%
                          </span>
                        </div>
                        <p className="text-white/60 mt-1 line-clamp-2">
                          {source.content.substring(0, 80)}...
                        </p>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </div>
            
            {/* Message tail */}
            <div
              className={`absolute bottom-0 w-4 h-4 ${
                isUser
                  ? 'right-0 translate-x-1 bg-emerald-600'
                  : 'left-0 -translate-x-1 bg-gray-800'
              } transform rotate-45 rounded-sm`}
            />
          </motion.div>
          
          <div className={`flex items-center space-x-2 mt-1 px-1 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
            <span className="text-xs text-gray-400">
              {formatTime(message.timestamp)}
            </span>
            {isUser && <MessageStatus status={message.status} />}
            {message.metadata?.processing_time && (
              <span className="text-xs text-gray-500 flex items-center">
                <FiCpu className="w-3 h-3 mr-1" />
                {Math.round(message.metadata.processing_time)}ms
              </span>
            )}
          </div>
        </div>
        
        {isUser && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-indigo-500 flex items-center justify-center shadow-lg"
          >
            <FiUser className="w-5 h-5 text-white" />
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

const ChatHeader = ({ 
  isOnline, 
  isTyping, 
  stats, 
  onClear, 
  onRefresh 
}: {
  isOnline: boolean;
  isTyping: boolean;
  stats: ChatStats;
  onClear: () => void;
  onRefresh: () => void;
}) => {
  const [showMenu, setShowMenu] = useState(false);

  return (
    <motion.div
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="relative bg-gradient-to-r from-gray-900/90 to-gray-800/90 backdrop-blur-xl border-b border-gray-700/50 p-4"
    >
      {/* Glass morphism overlay */}
      <div className="absolute inset-0 bg-white/5 backdrop-blur-sm" />
      
      <div className="relative z-10 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          {/* AI Avatar with status */}
          <div className="relative">
            <motion.div
              animate={{
                scale: isTyping ? [1, 1.1, 1] : 1,
                rotate: isTyping ? [0, 5, -5, 0] : 0
              }}
              transition={{
                duration: 2,
                repeat: isTyping ? Infinity : 0
              }}
              className="w-12 h-12 rounded-full bg-gradient-to-br from-emerald-400 via-cyan-400 to-blue-500 p-0.5 shadow-lg"
            >
              <div className="w-full h-full rounded-full bg-gray-900 flex items-center justify-center">
                <FiZap className="w-6 h-6 text-emerald-400" />
              </div>
            </motion.div>
            
            {/* Status indicator */}
            <motion.div
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.7, 1, 0.7]
              }}
              transition={{
                duration: 2,
                repeat: Infinity
              }}
              className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-gray-900 ${
                isTyping ? 'bg-yellow-400' : isOnline ? 'bg-emerald-400' : 'bg-gray-400'
              }`}
            />
          </div>
          
          <div>
            <h1 className="text-lg font-semibold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
              Assistente IA Avançado
            </h1>
            <div className="flex items-center space-x-4 text-sm text-gray-400">
              <div className="flex items-center space-x-1">
                {isOnline ? <FiWifi className="w-4 h-4" /> : <FiWifiOff className="w-4 h-4" />}
                <span>{isTyping ? 'Digitando...' : isOnline ? 'Online' : 'Offline'}</span>
              </div>
              <div className="flex items-center space-x-1">
                <FiMessageSquare className="w-4 h-4" />
                <span>{stats.totalMessages} mensagens</span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Actions */}
        <div className="flex items-center space-x-2">
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={onRefresh}
            className="p-2 rounded-lg bg-gray-700/50 hover:bg-gray-600/50 transition-colors"
            title="Atualizar"
          >
            <FiRefreshCw className="w-5 h-5 text-gray-300" />
          </motion.button>
          
          <div className="relative">
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setShowMenu(!showMenu)}
              className="p-2 rounded-lg bg-gray-700/50 hover:bg-gray-600/50 transition-colors"
            >
              <FiMoreVertical className="w-5 h-5 text-gray-300" />
            </motion.button>
            
            <AnimatePresence>
              {showMenu && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95, y: -10 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95, y: -10 }}
                  className="absolute right-0 top-12 w-48 bg-gray-800/95 backdrop-blur-sm border border-gray-600/50 rounded-xl shadow-2xl z-50 overflow-hidden"
                >
                  <motion.button
                    whileHover={{ backgroundColor: 'rgba(55, 65, 81, 0.5)' }}
                    onClick={() => {
                      onClear();
                      setShowMenu(false);
                    }}
                    className="w-full px-4 py-3 text-left text-sm text-white flex items-center space-x-3 transition-colors"
                  >
                    <FiTrash2 className="w-4 h-4 text-red-400" />
                    <span>Limpar conversa</span>
                  </motion.button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
      
      {/* Animated background particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(5)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-emerald-400/30 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -20, 0],
              opacity: [0.3, 0.7, 0.3],
              scale: [1, 1.5, 1]
            }}
            transition={{
              duration: 3 + Math.random() * 2,
              repeat: Infinity,
              delay: Math.random() * 2
            }}
          />
        ))}
      </div>
    </motion.div>
  );
};

const MessageInput = ({
  value,
  onChange,
  onSend,
  onFileSelect,
  disabled,
  isOnline
}: {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onFileSelect: (file: File) => void;
  disabled: boolean;
  isOnline: boolean;
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="relative bg-gradient-to-r from-gray-900/90 to-gray-800/90 backdrop-blur-xl border-t border-gray-700/50 p-4"
    >
      {/* Glass morphism overlay */}
      <div className="absolute inset-0 bg-white/5 backdrop-blur-sm" />
      
      <div className="relative z-10">
        <div className="relative flex items-end space-x-3">
          {/* File upload */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            className="flex-shrink-0 p-3 rounded-xl bg-gray-700/50 hover:bg-gray-600/50 text-gray-300 hover:text-emerald-400 transition-all disabled:opacity-50 disabled:cursor-not-allowed border border-gray-600/30"
            title="Anexar arquivo"
          >
            <FiPaperclip className="w-5 h-5" />
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              onChange={handleFileChange}
              accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.gif"
              disabled={disabled}
            />
          </motion.button>
          
          {/* Message input */}
          <div className="flex-1 relative">
            <textarea
              value={value}
              onChange={(e) => onChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={isOnline ? "Digite sua mensagem..." : "Modo offline - mensagens serão salvas localmente"}
              disabled={disabled}
              rows={1}
              className="w-full bg-gray-700/50 border border-gray-600/50 rounded-xl px-4 py-3 pr-12 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 resize-none backdrop-blur-sm transition-all disabled:opacity-50"
              style={{ minHeight: '48px', maxHeight: '120px' }}
            />
            
            {/* Send button */}
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={onSend}
              disabled={disabled || !value.trim()}
              className="absolute right-2 bottom-2 p-2 rounded-lg bg-gradient-to-r from-emerald-500 to-cyan-500 text-white disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed transition-all shadow-lg"
            >
              <FiSend className="w-4 h-4" />
            </motion.button>
          </div>
        </div>
        
        {/* Status text */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-xs text-gray-500 mt-2 text-center"
        >
          {!isOnline ? (
            <span className="text-yellow-400">Modo offline ativado</span>
          ) : (
            'O assistente pode cometer erros. Verifique informações importantes.'
          )}
        </motion.p>
      </div>
    </motion.div>
  );
};

// Main Chat Interface Component
const ModernChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Olá! Sou seu assistente de IA avançado. Como posso ajudá-lo hoje? 🚀',
      role: 'assistant',
      timestamp: new Date(),
      status: 'delivered'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isOnline, setIsOnline] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const stats: ChatStats = {
    totalMessages: messages.length,
    avgResponseTime: 1200,
    tokensUsed: 1500,
    documentsProcessed: 3
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      role: 'user',
      timestamp: new Date(),
      status: 'sending'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    // Simulate API call
    setTimeout(() => {
      setMessages(prev => 
        prev.map(msg => 
          msg.id === userMessage.id 
            ? { ...msg, status: 'delivered' }
            : msg
        )
      );

      // Simulate AI response
      setTimeout(() => {
        const aiResponse: Message = {
          id: (Date.now() + 1).toString(),
          content: 'Esta é uma resposta simulada do assistente de IA. Em uma implementação real, esta seria a resposta processada pelo modelo de linguagem.',
          role: 'assistant',
          timestamp: new Date(),
          status: 'delivered',
          metadata: {
            processing_time: 1200,
            tokens_used: 150,
            sources: [
              {
                content: 'Este é um exemplo de fonte consultada pelo assistente.',
                score: 0.95,
                metadata: {
                  filename: 'documento_exemplo.pdf',
                  page: 1
                }
              }
            ]
          }
        };

        setMessages(prev => [...prev, aiResponse]);
        setIsTyping(false);
      }, 2000);
    }, 1000);
  };

  const handleFileSelect = (file: File) => {
    const attachment: Attachment = {
      id: Date.now().toString(),
      name: file.name,
      type: 'document', // Simplificado para o exemplo
      size: file.size,
      progress: 0
    };

    const userMessage: Message = {
      id: Date.now().toString(),
      content: '',
      role: 'user',
      timestamp: new Date(),
      status: 'sending',
      attachments: [attachment]
    };

    setMessages(prev => [...prev, userMessage]);

    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setMessages(prev => 
        prev.map(msg => {
          if (msg.id === userMessage.id && msg.attachments) {
            return {
              ...msg,
              attachments: msg.attachments.map(att => ({
                ...att,
                progress: Math.min((att.progress || 0) + 10, 100)
              }))
            };
          }
          return msg;
        })
      );
    }, 200);

    setTimeout(() => {
      clearInterval(progressInterval);
      setMessages(prev => 
        prev.map(msg => 
          msg.id === userMessage.id 
            ? { ...msg, status: 'delivered' }
            : msg
        )
      );
    }, 3000);
  };

  const handleClearChat = () => {
    setMessages([
      {
        id: '1',
        content: 'Conversa limpa! Como posso ajudá-lo? 🔄',
        role: 'assistant',
        timestamp: new Date(),
        status: 'delivered'
      }
    ]);
  };

  const handleRefresh = () => {
    // Simulate refresh
    console.log('Refreshing chat...');
  };

  return (
    <div className="h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 rounded-full bg-emerald-900/10 blur-3xl animate-pulse" />
        <div className="absolute bottom-1/3 right-1/3 w-96 h-96 rounded-full bg-blue-900/10 blur-3xl animate-pulse delay-700" />
        <div className="absolute top-1/3 right-1/4 w-48 h-48 rounded-full bg-purple-900/10 blur-2xl animate-pulse delay-1000" />
      </div>

      <div className="relative z-10 h-full flex flex-col max-w-6xl mx-auto">
        {/* Header */}
        <ChatHeader
          isOnline={isOnline}
          isTyping={isTyping}
          stats={stats}
          onClear={handleClearChat}
          onRefresh={handleRefresh}
        />

        {/* Messages */}
        <div className="flex-1 overflow-hidden relative">
          <ScrollArea className="h-full px-4 py-6">
            <motion.div
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              className="space-y-2"
            >
              <AnimatePresence>
                {messages.map((message) => (
                  <MessageBubble key={message.id} message={message} />
                ))}
                {isTyping && (
                  <motion.div
                    key="typing"
                    variants={messageVariants}
                    initial="hidden"
                    animate="visible"
                    exit="exit"
                    className="flex justify-start mb-4"
                  >
                    <div className="flex items-end space-x-3">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center shadow-lg">
                        <RiRobotLine className="w-5 h-5 text-white" />
                      </div>
                      <div className="bg-gradient-to-br from-gray-700 to-gray-800 rounded-2xl rounded-bl-md px-4 py-3 border border-gray-600/50">
                        <TypingIndicator />
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
            <div ref={messagesEndRef} />
          </ScrollArea>
        </div>

        {/* Input */}
        <MessageInput
          value={inputValue}
          onChange={setInputValue}
          onSend={handleSendMessage}
          onFileSelect={handleFileSelect}
          disabled={isTyping}
          isOnline={isOnline}
        />
      </div>
    </div>
  );
};

export default ModernChatInterface;
