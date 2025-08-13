'use client';

import React, { useState, useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useRouter } from 'next/navigation';
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
import { ChatLogger, useChatLogger } from '@/app/chat/registroschatELK';
import {
    saveMessageToCache,
    loadCachedMessages,
    removeFromCache,
    checkConnectionStatus,
    setupConnectionListener,
    processFileForOffline,
    clearAllChatCache
} from '@/app/chat/chatCache';
import { aiClient, type RAGRequest, type RAGResponse, AIServiceError } from '@/lib/ai-client';
import { AILoadingIndicator, BotTypingIndicator, useLoadingStates } from '@/components/ui/loading';

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
  extension?: string;
}

interface Source {
  content?: string;
  score?: number;
  metadata?: {
    filename?: string;
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

// Animation variants
const messageVariants = {
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
      type: "spring" as const,
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

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const typingVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring" as const,
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

// Função para gerar IDs únicos
const generateUniqueId = () => {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    
    const timestamp = Date.now();
    const random = Math.random().toString(36).substr(2, 9);
    const performance = typeof window !== 'undefined' && window.performance 
        ? window.performance.now().toString(36).replace('.', '') 
        : '';
    const increment = (Date.now() % 1000000).toString(36);
    
    return `${timestamp}-${random}-${performance}-${increment}`;
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
              {attachment.extension?.toUpperCase() || attachment.type.toUpperCase()} • {formatFileSize(attachment.size)}
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
        className="flex justify-center my-6"
      >
        <div className="bg-gradient-to-r from-yellow-500/15 to-orange-500/15 border border-yellow-500/25 rounded-2xl px-6 py-3 max-w-md backdrop-blur-sm shadow-lg">
          <div className="flex items-center space-x-3">
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            >
              <FiAlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0" />
            </motion.div>
            <p className="text-sm text-yellow-300 font-medium">{message.content}</p>
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
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}
    >
      <div className="flex items-end space-x-3 max-w-[85%]">
        {!isUser && (
          <motion.div
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            className="flex-shrink-0 w-11 h-11 rounded-full bg-gradient-to-br from-emerald-400 via-cyan-400 to-blue-500 p-0.5 shadow-xl"
          >
            <div className="w-full h-full rounded-full bg-gray-900 flex items-center justify-center">
              <RiRobotLine className="w-6 h-6 text-emerald-400" />
            </div>
          </motion.div>
        )}
        
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
          <motion.div
            whileHover={{ scale: 1.02 }}
            className={`relative rounded-3xl px-5 py-4 max-w-lg shadow-2xl ${
              isUser
                ? 'bg-gradient-to-br from-emerald-500 to-emerald-600 text-white rounded-br-lg'
                : 'bg-gradient-to-br from-gray-700/90 to-gray-800/90 text-white rounded-bl-lg border border-gray-600/40'
            }`}
          >
            {/* Enhanced glass morphism effect */}
            <div className={`absolute inset-0 rounded-3xl backdrop-blur-sm ${
              isUser ? 'bg-white/10' : 'bg-white/5'
            }`} />
            
            {/* Subtle glow effect */}
            <div className={`absolute inset-0 rounded-3xl ${
              isUser 
                ? 'bg-gradient-to-br from-emerald-400/20 to-emerald-600/20' 
                : 'bg-gradient-to-br from-gray-600/10 to-gray-800/10'
            }`} />
            
            <div className="relative z-10">
              {message.content && (
                <p className="text-sm leading-relaxed whitespace-pre-wrap font-medium">
                  {message.content}
                </p>
              )}
              
              {message.attachments && message.attachments.length > 0 && (
                <div className="mt-4 space-y-3">
                  {message.attachments.map((attachment) => (
                    <AttachmentCard key={attachment.id} attachment={attachment} />
                  ))}
                </div>
              )}
            </div>
            
            {/* Enhanced message tail */}
            <div
              className={`absolute bottom-0 w-5 h-5 ${
                isUser
                  ? 'right-0 translate-x-1 bg-emerald-600'
                  : 'left-0 -translate-x-1 bg-gray-800'
              } transform rotate-45 rounded-sm border ${
                isUser ? 'border-emerald-500' : 'border-gray-600/40'
              }`}
            />
          </motion.div>
          
          <div className={`flex items-center space-x-3 mt-2 px-2 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
            <span className="text-xs text-gray-400 font-medium">
              {formatTime(message.timestamp)}
            </span>
            {isUser && <MessageStatus status={message.status} />}
            {message.metadata?.processing_time && (
              <span className="text-xs text-gray-500 flex items-center bg-gray-800/40 rounded-full px-2 py-1">
                <FiCpu className="w-3 h-3 mr-1" />
                {Math.round(message.metadata.processing_time)}ms
              </span>
            )}
          </div>
        </div>
        
        {isUser && (
          <motion.div
            initial={{ scale: 0, rotate: 180 }}
            animate={{ scale: 1, rotate: 0 }}
            className="flex-shrink-0 w-11 h-11 rounded-full bg-gradient-to-br from-blue-400 via-indigo-500 to-purple-500 p-0.5 shadow-xl"
          >
            <div className="w-full h-full rounded-full bg-gray-900 flex items-center justify-center">
              <FiUser className="w-6 h-6 text-blue-400" />
            </div>
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
  onRefresh,
  onClearCache 
}: {
  isOnline: boolean;
  isTyping: boolean;
  stats: ChatStats;
  onClear: () => void;
  onRefresh: () => void;
  onClearCache: () => void;
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
          {/* Clear Cache Button */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={onClearCache}
            className="p-2.5 rounded-lg bg-gradient-to-r from-purple-600/20 to-purple-700/20 hover:from-purple-600/30 hover:to-purple-700/30 border border-purple-500/30 transition-all duration-200 group"
            title="Limpar Cache"
          >
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              className="group-hover:pause"
            >
              <FiActivity className="w-4 h-4 text-purple-400" />
            </motion.div>
          </motion.button>

          {/* Clear Chat Button */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={onClear}
            className="p-2.5 rounded-lg bg-gradient-to-r from-red-600/20 to-red-700/20 hover:from-red-600/30 hover:to-red-700/30 border border-red-500/30 transition-all duration-200"
            title="Limpar Conversa"
          >
            <FiTrash2 className="w-4 h-4 text-red-400" />
          </motion.button>
          
          {/* Refresh Button */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={onRefresh}
            className="p-2.5 rounded-lg bg-gradient-to-r from-emerald-600/20 to-emerald-700/20 hover:from-emerald-600/30 hover:to-emerald-700/30 border border-emerald-500/30 transition-all duration-200"
            title="Atualizar"
          >
            <FiRefreshCw className="w-4 h-4 text-emerald-400" />
          </motion.button>
          
          {/* More Options Button */}
          <div className="relative">
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setShowMenu(!showMenu)}
              className="p-2.5 rounded-lg bg-gradient-to-r from-gray-600/20 to-gray-700/20 hover:from-gray-600/30 hover:to-gray-700/30 border border-gray-500/30 transition-all duration-200"
              title="Mais opções"
            >
              <FiMoreVertical className="w-4 h-4 text-gray-300" />
            </motion.button>
            
            <AnimatePresence>
              {showMenu && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95, y: -10 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95, y: -10 }}
                  className="absolute right-0 top-12 w-52 bg-gray-800/95 backdrop-blur-sm border border-gray-600/50 rounded-xl shadow-2xl z-50 overflow-hidden"
                >
                  <motion.button
                    whileHover={{ backgroundColor: 'rgba(55, 65, 81, 0.5)' }}
                    onClick={() => {
                      onClear();
                      setShowMenu(false);
                    }}
                    className="w-full px-4 py-3 text-left text-sm text-white flex items-center space-x-3 transition-colors hover:bg-red-500/10"
                  >
                    <FiTrash2 className="w-4 h-4 text-red-400" />
                    <span>Limpar conversa</span>
                  </motion.button>
                  
                  <motion.button
                    whileHover={{ backgroundColor: 'rgba(55, 65, 81, 0.5)' }}
                    onClick={() => {
                      onClearCache();
                      setShowMenu(false);
                    }}
                    className="w-full px-4 py-3 text-left text-sm text-white flex items-center space-x-3 transition-colors hover:bg-purple-500/10 border-t border-gray-700/50"
                  >
                    <FiActivity className="w-4 h-4 text-purple-400" />
                    <span>Limpar cache do chat</span>
                  </motion.button>
                  
                  <motion.button
                    whileHover={{ backgroundColor: 'rgba(55, 65, 81, 0.5)' }}
                    onClick={() => {
                      onRefresh();
                      setShowMenu(false);
                    }}
                    className="w-full px-4 py-3 text-left text-sm text-white flex items-center space-x-3 transition-colors hover:bg-emerald-500/10 border-t border-gray-700/50"
                  >
                    <FiRefreshCw className="w-4 h-4 text-emerald-400" />
                    <span>Atualizar chat</span>
                  </motion.button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
      
      {/* Animated background particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-gradient-to-r from-emerald-400/20 to-cyan-400/20 rounded-full blur-sm"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -15, 0],
              x: [0, Math.random() * 10 - 5, 0],
              opacity: [0.1, 0.3, 0.1],
              scale: [1, 1.2, 1]
            }}
            transition={{
              duration: 6 + Math.random() * 4,
              repeat: Infinity,
              delay: Math.random() * 5,
              ease: "easeInOut"
            }}
          />
        ))}
        
        {/* Floating geometric shapes */}
        {[...Array(3)].map((_, i) => (
          <motion.div
            key={`shape-${i}`}
            className="absolute w-2 h-2 border border-emerald-400/10"
            style={{
              left: `${20 + i * 30}%`,
              top: `${30 + i * 20}%`,
              transform: 'rotate(45deg)'
            }}
            animate={{
              y: [0, -10, 0],
              rotate: [45, 90, 45],
              opacity: [0.05, 0.15, 0.05]
            }}
            transition={{
              duration: 10 + i * 3,
              repeat: Infinity,
              delay: i * 2,
              ease: "linear"
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
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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

  // Auto resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = Math.min(scrollHeight, 120) + 'px';
    }
  }, [value]);

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="relative bg-gradient-to-r from-gray-900/95 to-gray-800/95 backdrop-blur-xl border-t border-gray-700/50 p-4"
    >
      {/* Glass morphism overlay */}
      <div className="absolute inset-0 bg-white/3 backdrop-blur-sm" />
      
      {/* Glow effect */}
      <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/5 via-transparent to-cyan-500/5" />
      
      <div className="relative z-10">
        <div className="relative flex items-end space-x-3 max-w-4xl mx-auto">
          {/* File upload */}
          <motion.button
            whileHover={{ scale: 1.05, rotate: 5 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            className="flex-shrink-0 p-3 rounded-xl bg-gradient-to-r from-gray-700/60 to-gray-600/60 hover:from-gray-600/70 hover:to-gray-500/70 text-gray-300 hover:text-emerald-400 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed border border-gray-600/40 hover:border-emerald-500/40 shadow-lg hover:shadow-emerald-500/20"
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
          
          {/* Message input container */}
          <div className="flex-1 relative">
            <div className="relative">
              <textarea
                ref={textareaRef}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={isOnline ? "Digite sua mensagem..." : "Modo offline - mensagens serão salvas localmente"}
                disabled={disabled}
                rows={1}
                className="w-full bg-gradient-to-r from-gray-700/60 to-gray-600/60 border border-gray-600/50 rounded-xl px-4 py-3 pr-14 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 resize-none backdrop-blur-sm transition-all duration-300 disabled:opacity-50 hover:border-gray-500/60 shadow-inner"
                style={{ 
                  minHeight: '48px', 
                  maxHeight: '120px',
                  lineHeight: '1.5'
                }}
              />
              
              {/* Input glow effect */}
              <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 opacity-0 transition-opacity duration-300 group-focus-within:opacity-100 pointer-events-none" />
              
              {/* Send button */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={onSend}
                disabled={disabled || !value.trim()}
                className="absolute right-2 bottom-2 p-2.5 rounded-lg bg-gradient-to-r from-emerald-500 to-cyan-500 text-white disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-emerald-500/40 disabled:shadow-none"
              >
                <motion.div
                  animate={disabled || !value.trim() ? {} : {
                    rotate: [0, 10, -10, 0],
                  }}
                  transition={{
                    duration: 0.5,
                    ease: "easeInOut"
                  }}
                >
                  <FiSend className="w-4 h-4" />
                </motion.div>
              </motion.button>
            </div>
          </div>
        </div>
        
        {/* Status text */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-xs text-center mt-3 max-w-4xl mx-auto"
        >
          {!isOnline ? (
            <div className="flex items-center justify-center space-x-2">
              <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
              <span className="text-yellow-400 font-medium">Modo offline ativado</span>
            </div>
          ) : (
            <p className="text-gray-500">
              O assistente pode cometer erros. Verifique informações importantes.
            </p>
          )}
        </motion.div>
      </div>
      
      {/* Subtle background animation */}
      <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-emerald-500/20 to-transparent" />
    </motion.div>
  );
};

// Memoized background component to prevent re-renders during typing
const AnimatedBackground = React.memo(() => (
  <div className="absolute inset-0 overflow-hidden pointer-events-none">
    {/* Large gradient orbs */}
    <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-gradient-to-r from-emerald-900/10 to-emerald-600/8 blur-3xl animate-pulse" style={{animationDuration: '4s'}} />
    <div className="absolute bottom-1/3 right-1/3 w-[32rem] h-[32rem] rounded-full bg-gradient-to-r from-blue-900/8 to-cyan-600/6 blur-3xl animate-pulse delay-700" style={{animationDuration: '5s'}} />
    <div className="absolute top-1/3 right-1/4 w-64 h-64 rounded-full bg-gradient-to-r from-purple-900/8 to-purple-600/6 blur-2xl animate-pulse delay-1000" style={{animationDuration: '6s'}} />
    
    {/* Floating particles */}
    {[...Array(8)].map((_, i) => (
      <motion.div
        key={`particle-${i}`}
        className="absolute w-1 h-1 bg-gradient-to-r from-emerald-400/30 to-cyan-400/30 rounded-full"
        style={{
          left: `${Math.random() * 100}%`,
          top: `${Math.random() * 100}%`,
        }}
        animate={{
          y: [0, -20, 0],
          x: [0, Math.random() * 10 - 5, 0],
          opacity: [0.1, 0.4, 0.1],
          scale: [1, 1.3, 1]
        }}
        transition={{
          duration: 8 + Math.random() * 6,
          repeat: Infinity,
          delay: Math.random() * 8,
          ease: "easeInOut"
        }}
      />
    ))}
    
    {/* Geometric shapes */}
    {[...Array(6)].map((_, i) => (
      <motion.div
        key={`geo-${i}`}
        className="absolute border border-emerald-400/10 rounded-lg"
        style={{
          left: `${15 + i * 15}%`,
          top: `${20 + i * 12}%`,
          width: `${8 + i * 2}px`,
          height: `${8 + i * 2}px`,
        }}
        animate={{
          rotate: [0, 360],
          scale: [1, 1.1, 1],
          opacity: [0.05, 0.15, 0.05]
        }}
        transition={{
          duration: 15 + i * 5,
          repeat: Infinity,
          delay: i * 3,
          ease: "linear"
        }}
      />
    ))}
    
    {/* Neural network effect */}
    <svg className="absolute inset-0 w-full h-full opacity-[0.01]">
      <defs>
        <pattern id="neural-grid" x="0" y="0" width="120" height="120" patternUnits="userSpaceOnUse">
          <circle cx="60" cy="60" r="1" fill="currentColor" className="text-emerald-400">
            <animate attributeName="opacity" values="0.1;0.3;0.1" dur="6s" repeatCount="indefinite" />
          </circle>
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#neural-grid)" />
    </svg>
  </div>
));

AnimatedBackground.displayName = 'AnimatedBackground';

// Main Enhanced Chat Interface Component
const EnhancedChatInterface = () => {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isOnline, setIsOnline] = useState(true);
  const [isClient, setIsClient] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Using logger hook
  const { logs, addLog, addError, addWarning, addInfo, addRequest, addResponse, clearLogs } = useChatLogger();
  
  // Hook for loading states
  const { isLoading, withLoading } = useLoadingStates();

  const stats: ChatStats = {
    totalMessages: messages.length,
    avgResponseTime: 1200,
    tokensUsed: 1500,
    documentsProcessed: 3
  };

  // Function to safely add messages, avoiding duplicates
  const addMessage = (newMessage: Message) => {
    setMessages((prev) => {
      if (prev.some(msg => msg.id === newMessage.id)) {
        console.warn(`Message with ID ${newMessage.id} already exists, ignoring duplicate`);
        return prev;
      }
      return [...prev, newMessage];
    });
  };

  // Function to update existing message
  const updateMessage = (messageId: string, updater: (msg: Message) => Message) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId ? updater(msg) : msg
    ));
  };

  useEffect(() => {
    setIsClient(true);

    // Check initial connection status
    setIsOnline(checkConnectionStatus());

    // Load cached messages
    const loadCachedData = async () => {
      try {
        const cachedMessages = await loadCachedMessages();
        if (cachedMessages.length > 0) {
          // Convert to new message format
          const convertedMessages = cachedMessages.map(msg => ({
            id: msg.id,
            content: msg.text,
            role: msg.sender === 'user' ? 'user' as const : 'assistant' as const,
            timestamp: msg.timestamp,
            status: 'delivered' as const,
            attachments: msg.attachments?.map(att => ({
              ...att,
              type: att.type as Attachment['type']
            }))
          }));
          setMessages(convertedMessages);
          addInfo('Messages loaded from offline cache', { count: cachedMessages.length });
        } else {
          setMessages([{
            id: '1',
            content: 'Olá! Sou seu assistente de IA avançado. Como posso ajudá-lo hoje? 🚀',
            role: 'assistant',
            timestamp: new Date(),
            status: 'delivered'
          }]);
        }
      } catch (error) {
        addError('Error loading messages from cache', {
          error: error instanceof Error ? error.message : 'Unknown error'
        });
        setMessages([{
          id: '1',
          content: 'Olá! Sou seu assistente de IA avançado. Como posso ajudá-lo hoje? 🚀',
          role: 'assistant',
          timestamp: new Date(),
          status: 'delivered'
        }]);
      }
    };

    loadCachedData();
    addInfo('Chat session started');

    // Setup connection listener
    const cleanupListener = setupConnectionListener((online) => {
      setIsOnline(online);
      addInfo(online ? 'Connection restored' : 'Offline mode activated');

      if (online) {
        // Try to sync pending messages when connection returns
        // (Implement sync logic as needed)
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
      addWarning('User not authenticated, redirecting to login');
      router.push('/login');
    } else {
      addInfo('User authenticated, chat ready for use');
    }
  }, [router, isClient]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: generateUniqueId(),
      content: inputValue,
      role: 'user',
      timestamp: new Date(),
      status: 'sending'
    };

    addRequest('User message', {
      messageId: userMessage.id,
      messageLength: userMessage.content.length,
      truncatedText: userMessage.content.length > 50
        ? userMessage.content.substring(0, 50) + '...'
        : userMessage.content
    });

    // Save to cache immediately
    await saveMessageToCache({
      id: userMessage.id,
      text: userMessage.content,
      sender: 'user',
      timestamp: userMessage.timestamp
    });
    
    addMessage(userMessage);
    setInputValue('');
    setIsTyping(true);

    // Update message status to sent
    setTimeout(() => {
      updateMessage(userMessage.id, (msg) => ({ ...msg, status: 'sent' }));
    }, 500);

    try {
      if (!isOnline) {
        // Offline mode - simulate bot response
        await new Promise(resolve => setTimeout(resolve, 1000));

        const botMessage: Message = {
          id: generateUniqueId(),
          content: 'Estou offline no momento. Sua mensagem foi salva e será processada quando a conexão for restaurada.',
          role: 'assistant',
          timestamp: new Date(),
          status: 'delivered'
        };

        await saveMessageToCache({
          id: botMessage.id,
          text: botMessage.content,
          sender: 'bot',
          timestamp: botMessage.timestamp
        });
        
        addMessage(botMessage);
        addWarning('Offline mode - message saved locally');
        
        // Update user message status to delivered
        updateMessage(userMessage.id, (msg) => ({ ...msg, status: 'delivered' }));
        return;
      }

      // REAL AI INTEGRATION
      const ragRequest: RAGRequest = {
        question: userMessage.content,
        max_tokens: 512,
        temperature: 0.7,
        search_limit: 3,
        score_threshold: 0.6
      };

      const startTime = Date.now();
      addInfo('Sending question to AI...', { question: userMessage.content });

      const aiResponse: RAGResponse = await withLoading('ai-query', async () => {
        return await aiClient.askQuestion(ragRequest);
      });
      
      const endTime = Date.now();

      // Check if response is valid before processing
      if (!aiResponse || !aiResponse.answer) {
        throw new AIServiceError('Invalid response received from AI service');
      }

      const botMessage: Message = {
        id: generateUniqueId(),
        content: aiResponse.answer,
        role: 'assistant',
        timestamp: new Date(),
        status: 'delivered',
        metadata: {
          processing_time: aiResponse.processing_time || (endTime - startTime),
          tokens_used: aiResponse.tokens_used || 0,
          search_time: aiResponse.search_time || 0,
          generation_time: aiResponse.generation_time || 0,
          sources: aiResponse.sources?.map(source => ({
            content: source.content || '',
            score: source.score || 0,
            metadata: {
              filename: source.metadata?.filename || 'Unknown',
              page: source.metadata?.page,
              chunk_index: source.metadata?.chunk_index
            }
          }))
        }
      };

      addResponse('AI response received', {
        userMessageId: userMessage.id,
        botMessageId: botMessage.id,
        responseTimeMs: endTime - startTime,
        tokensUsed: aiResponse.tokens_used || 0,
        sourcesFound: aiResponse.sources?.length || 0,
        processingTime: aiResponse.processing_time || 0,
        searchTime: aiResponse.search_time || 0,
        generationTime: aiResponse.generation_time || 0
      });

      // Save message to cache and update state
      try {
        await saveMessageToCache({
          id: botMessage.id,
          text: botMessage.content,
          sender: 'bot',
          timestamp: botMessage.timestamp
        });
        
        addMessage(botMessage);

        // Update user message status to delivered
        updateMessage(userMessage.id, (msg) => ({ ...msg, status: 'delivered' }));

        // Log sources found for debugging
        if (aiResponse.sources && aiResponse.sources.length > 0) {
          addInfo('Sources used in response', {
            sources: aiResponse.sources.map(source => ({
              filename: source.metadata?.filename || 'Unknown',
              score: source.score || 0,
              contentPreview: source.content ? source.content.substring(0, 100) + '...' : 'No content'
            }))
          });
        }
      } catch (cacheError) {
        // If cache saving fails, still show message to user
        console.warn('Failed to save to cache, but message will be displayed:', cacheError);
        addMessage(botMessage);
        updateMessage(userMessage.id, (msg) => ({ ...msg, status: 'delivered' }));
        
        addWarning('Failed to save to cache', {
          error: cacheError instanceof Error ? cacheError.message : 'Unknown error'
        });
      }

    } catch (error) {
      console.error('Error during message processing:', error);
      
      let errorMessage = 'Unexpected error processing your message';
      let shouldDisplayError = true;
      
      if (error instanceof AIServiceError) {
        addError('AI service error', {
          service: error.service,
          status: error.status,
          message: error.message,
          inputMessage: userMessage.content
        });
        
        if (error.status === 503) {
          errorMessage = 'AI services are temporarily unavailable. Please try again in a few moments.';
        } else if (error.status === 429) {
          errorMessage = 'Too many requests. Please wait a moment before trying again.';
        } else if (error.status === 401) {
          errorMessage = 'Session expired. Please log in again.';
        } else if (error.status === 404) {
          errorMessage = 'Service not found. Please check if AI services are running.';
        } else {
          errorMessage = `AI service error: ${error.message}`;
        }
      } else if (error instanceof Error) {
        // Check if it's a known error that shouldn't display error message
        if (error.message.includes('AbortError') || error.message.includes('cancelled')) {
          // User cancelled operation, don't show error
          shouldDisplayError = false;
          addInfo('Operation cancelled by user');
        } else {
          addError('Failed to process message', {
            error: error.message,
            inputMessage: userMessage.content,
            stack: error.stack
          });
          
          if (error.message.includes('NetworkError') || error.message.includes('fetch')) {
            errorMessage = 'Connection error. Check your internet and try again.';
          }
        }
      } else {
        addError('Unknown error', {
          error: String(error),
          inputMessage: userMessage.content
        });
      }

      // Only show error message if necessary
      if (shouldDisplayError) {
        const errorBotMessage: Message = {
          id: generateUniqueId(),
          content: errorMessage,
          role: 'system',
          timestamp: new Date(),
          status: 'delivered'
        };

        try {
          await saveMessageToCache({
            id: errorBotMessage.id,
            text: errorBotMessage.content,
            sender: 'bot',
            timestamp: errorBotMessage.timestamp
          });
        } catch (cacheError) {
          console.warn('Failed to save error message to cache:', cacheError);
        }
        
        addMessage(errorBotMessage);
      }
      
      // Update user message status to error
      updateMessage(userMessage.id, (msg) => ({ ...msg, status: 'error' }));
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
    return 'text';
  };

  const getFileExtension = (fileName: string): string => {
    return fileName.split('.').pop()?.toLowerCase() || '';
  };

  const handleFileSelect = async (file: File) => {
    const fileExtension = getFileExtension(file.name);

    addInfo('File upload started', {
      fileName: file.name,
      fileType: file.type,
      fileSize: file.size,
      fileExtension
    });

    // Validate file type
    if (!ALLOWED_FILE_TYPES.includes(file.type)) {
      addError('Unsupported file type', {
        fileName: file.name,
        fileType: file.type,
        allowedTypes: ALLOWED_FILE_TYPES
      });

      const errorMessage: Message = {
        id: generateUniqueId(),
        content: `Erro: Tipo de arquivo não suportado (${fileExtension || file.type}). Formatos aceitos: PDF, DOC, DOCX, XLS, XLSX, PPT, PPTX, JPG, PNG, GIF, TXT`,
        role: 'system',
        timestamp: new Date(),
        status: 'delivered'
      };

      await saveMessageToCache({
        id: errorMessage.id,
        text: errorMessage.content,
        sender: 'bot',
        timestamp: errorMessage.timestamp
      });
      
      addMessage(errorMessage);
      return;
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
      addWarning('Upload rejected - file too large', {
        fileName: file.name,
        fileSize: file.size,
        maxAllowed: '10MB'
      });

      const errorMessage: Message = {
        id: generateUniqueId(),
        content: `Erro: O arquivo é muito grande (${formatFileSize(file.size)}). Tamanho máximo permitido: 10MB`,
        role: 'system',
        timestamp: new Date(),
        status: 'delivered'
      };

      await saveMessageToCache({
        id: errorMessage.id,
        text: errorMessage.content,
        sender: 'bot',
        timestamp: errorMessage.timestamp
      });
      
      addMessage(errorMessage);
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    // Process file for offline storage
    const processedFile = await processFileForOffline(file);

    const attachment: Attachment = {
      id: processedFile.id,
      name: processedFile.name,
      type: getFileType(file),
      size: processedFile.size,
      progress: 0,
      extension: processedFile.extension
    };

    const userMessage: Message = {
      id: generateUniqueId(),
      content: `Enviando arquivo: ${file.name}`,
      role: 'user',
      timestamp: new Date(),
      status: 'sending',
      attachments: [attachment]
    };

    // Save to cache immediately
    await saveMessageToCache({
      id: userMessage.id,
      text: userMessage.content,
      sender: 'user',
      timestamp: userMessage.timestamp,
      attachments: [{
        id: attachment.id,
        name: attachment.name,
        type: attachment.type,
        size: attachment.size,
        extension: attachment.extension
      }]
    });
    
    addMessage(userMessage);

    // Simulate upload progress
    const uploadInterval = setInterval(() => {
      setUploadProgress((prev) => {
        const newProgress = prev + Math.random() * 10;
        if (newProgress >= 100) {
          clearInterval(uploadInterval);
          return 100;
        }
        
        // Update attachment progress
        updateMessage(userMessage.id, (msg) => ({
          ...msg,
          attachments: msg.attachments?.map(att => ({
            ...att,
            progress: newProgress
          }))
        }));
        
        return newProgress;
      });
    }, 200);

    try {
      if (!isOnline) {
        // Offline mode - just store locally
        await new Promise(resolve => setTimeout(resolve, 1500));

        const botMessage: Message = {
          id: generateUniqueId(),
          content: 'Arquivo recebido em modo offline. Será processado quando a conexão for restaurada.',
          role: 'assistant',
          timestamp: new Date(),
          status: 'delivered'
        };

        await saveMessageToCache({
          id: botMessage.id,
          text: botMessage.content,
          sender: 'bot',
          timestamp: botMessage.timestamp
        });
        
        updateMessage(userMessage.id, (msg) => ({
          ...msg,
          status: 'delivered',
          content: 'Arquivo salvo localmente (modo offline)'
        }));
        
        addMessage(botMessage);
        addWarning('Offline mode - file saved locally');
        return;
      }

      // REAL UPLOAD INTEGRATION
      addInfo('Sending file for processing...', { 
        fileName: file.name, 
        fileSize: file.size 
      });

      const uploadResponse = await withLoading('file-upload', async () => {
        return await aiClient.uploadDocument(file);
      });

      if (uploadResponse.success) {
        addInfo('File upload completed successfully', {
          fileName: file.name,
          fileSize: file.size,
          processingTimeMs: Date.now() - userMessage.timestamp.getTime(),
          documentId: uploadResponse.document_id,
          chunksCreated: uploadResponse.chunks_created
        });

        const updatedMessage: Message = {
          ...userMessage,
          status: 'delivered',
          content: uploadResponse.message || 'Arquivo processado com sucesso'
        };

        await saveMessageToCache({
          id: updatedMessage.id,
          text: updatedMessage.content,
          sender: 'user',
          timestamp: updatedMessage.timestamp,
          attachments: updatedMessage.attachments?.map(att => ({
            id: att.id,
            name: att.name,
            type: att.type,
            size: att.size,
            extension: att.extension
          }))
        });
        
        updateMessage(userMessage.id, () => updatedMessage);

        // Automatic bot response about processed document
        const botMessage: Message = {
          id: generateUniqueId(),
          content: `✅ Documento "${file.name}" foi processado e indexado com sucesso! ${uploadResponse.chunks_created ? `Foram criados ${uploadResponse.chunks_created} fragmentos para busca.` : ''} Agora você pode fazer perguntas sobre o conteúdo deste documento.`,
          role: 'assistant',
          timestamp: new Date(),
          status: 'delivered'
        };

        await saveMessageToCache({
          id: botMessage.id,
          text: botMessage.content,
          sender: 'bot',
          timestamp: botMessage.timestamp
        });
        
        addMessage(botMessage);
      } else {
        throw new Error(uploadResponse.message || 'Error processing file');
      }
    } catch (error) {
      addError('File upload failed', {
        fileName: file.name,
        error: error instanceof Error ? error.message : 'Unknown error',
        fileType: file.type,
        fileSize: file.size
      });

      let errorMessage = 'Error processing file';
      
      if (error instanceof AIServiceError) {
        if (error.status === 413) {
          errorMessage = 'File too large. Maximum size: 10MB';
        } else if (error.status === 415) {
          errorMessage = 'Unsupported file type';
        } else if (error.status === 503) {
          errorMessage = 'Processing service unavailable. Please try again in a few moments.';
        } else {
          errorMessage = `Upload error: ${error.message}`;
        }
      } else if (error instanceof Error) {
        errorMessage = `Error processing file: ${error.message}`;
      }

      const errorBotMessage: Message = {
        id: generateUniqueId(),
        content: errorMessage,
        role: 'system',
        timestamp: new Date(),
        status: 'delivered'
      };

      await saveMessageToCache({
        id: errorBotMessage.id,
        text: errorBotMessage.content,
        sender: 'bot',
        timestamp: errorBotMessage.timestamp
      });
      
      addMessage(errorBotMessage);
      
      // Update user message status to error
      updateMessage(userMessage.id, (msg) => ({ ...msg, status: 'error' }));
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleClearChat = async () => {
    try {
      // Limpar mensagens localmente
      setMessages([{
        id: generateUniqueId(),
        content: 'Conversa limpa! Como posso ajudá-lo? 🔄',
        role: 'assistant',
        timestamp: new Date(),
        status: 'delivered'
      }]);
      addInfo('Chat messages cleared successfully');
    } catch (error) {
      addError('Error clearing messages', {
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  };

  const handleClearCache = async () => {
    try {
      // Limpar cache do chat
      await clearAllChatCache();
      
      // Resetar mensagens para o estado inicial
      setMessages([{
        id: generateUniqueId(),
        content: 'Cache limpo com sucesso! Todas as mensagens offline foram removidas. Como posso ajudá-lo? 🗑️✨',
        role: 'assistant',
        timestamp: new Date(),
        status: 'delivered'
      }]);
      
      addInfo('Chat cache cleared successfully');
    } catch (error) {
      addError('Error clearing cache', {
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      
      // Mostrar mensagem de erro para o usuário
      const errorMessage: Message = {
        id: generateUniqueId(),
        content: 'Erro ao limpar o cache. Tente novamente.',
        role: 'system',
        timestamp: new Date(),
        status: 'delivered'
      };
      
      addMessage(errorMessage);
    }
  };

  const handleRefresh = () => {
    // Implement refresh logic
    addInfo('Refreshing chat...');
  };

  if (!isClient) {
    return null;
  }

  return (
    <div className="h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white relative overflow-hidden">
      {/* Enhanced animated background */}
      <AnimatedBackground />

      <div className="relative z-10 h-full flex flex-col max-w-6xl mx-auto">
        {/* Header */}
        <ChatHeader
          isOnline={isOnline}
          isTyping={isTyping}
          stats={stats}
          onClear={handleClearChat}
          onRefresh={handleRefresh}
          onClearCache={handleClearCache}
        />

        {/* Offline notification */}
        {!isOnline && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gradient-to-r from-yellow-500/20 to-orange-500/20 text-yellow-300 text-sm p-3 text-center border-b border-yellow-500/30 backdrop-blur-sm"
          >
            <div className="flex items-center justify-center space-x-2">
              <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
              <span>Você está offline. As mensagens serão salvas localmente e enviadas quando a conexão for restaurada.</span>
            </div>
          </motion.div>
        )}

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
          disabled={isTyping || isUploading}
          isOnline={isOnline}
        />
      </div>

      {/* Logger */}
      <ChatLogger logs={logs} floating />
    </div>
  );
};

export default EnhancedChatInterface;
