/**
 * Componentes de loading states para diferentes operações da IA
 */

'use client';

import { useState } from "react";

interface LoadingProps {
  type?: 'thinking' | 'uploading' | 'processing' | 'searching';
  message?: string;
  progress?: number;
  className?: string;
}

export function AILoadingIndicator({ 
  type = 'thinking', 
  message,
  progress,
  className = '' 
}: LoadingProps) {
  const getLoadingContent = () => {
    switch (type) {
      case 'thinking':
        return {
          icon: (
            <div className="flex space-x-1">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-bounce"></div>
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: '0.4s' }}></div>
            </div>
          ),
          text: message || 'Pensando...',
          description: 'Analisando sua pergunta e buscando informações relevantes'
        };
      
      case 'uploading':
        return {
          icon: (
            <div className="relative">
              <svg className="w-6 h-6 text-blue-400 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle 
                  className="opacity-25" 
                  cx="12" 
                  cy="12" 
                  r="10" 
                  stroke="currentColor" 
                  strokeWidth="4"
                />
                <path 
                  className="opacity-75" 
                  fill="currentColor" 
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              {progress !== undefined && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-xs font-bold text-blue-400">{Math.round(progress)}%</span>
                </div>
              )}
            </div>
          ),
          text: message || 'Enviando arquivo...',
          description: progress !== undefined ? `${Math.round(progress)}% concluído` : 'Fazendo upload do seu documento'
        };
      
      case 'processing':
        return {
          icon: (
            <div className="relative">
              <div className="w-6 h-6 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
              <div className="absolute inset-1 border border-purple-300 border-t-transparent rounded-full animate-spin animate-reverse"></div>
            </div>
          ),
          text: message || 'Processando documento...',
          description: 'Extraindo texto e criando índices para busca'
        };
      
      case 'searching':
        return {
          icon: (
            <div className="flex space-x-1">
              <div className="w-1 h-4 bg-yellow-400 animate-pulse"></div>
              <div className="w-1 h-4 bg-yellow-400 animate-pulse" style={{ animationDelay: '0.1s' }}></div>
              <div className="w-1 h-4 bg-yellow-400 animate-pulse" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-1 h-4 bg-yellow-400 animate-pulse" style={{ animationDelay: '0.3s' }}></div>
              <div className="w-1 h-4 bg-yellow-400 animate-pulse" style={{ animationDelay: '0.4s' }}></div>
            </div>
          ),
          text: message || 'Buscando informações...',
          description: 'Procurando nos documentos indexados'
        };
    }
  };

  const content = getLoadingContent();

  return (
    <div className={`flex items-center space-x-3 p-3 rounded-lg bg-gray-700/50 backdrop-blur-sm border border-gray-600/30 ${className}`}>
      <div className="flex-shrink-0">
        {content.icon}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-white">
          {content.text}
        </p>
        <p className="text-xs text-gray-400">
          {content.description}
        </p>
        {progress !== undefined && (
          <div className="w-full bg-gray-600 rounded-full h-1.5 mt-2">
            <div 
              className="bg-gradient-to-r from-blue-500 to-blue-600 h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

// Componente específico para mensagens de typing do bot
export function BotTypingIndicator({ className = '' }: { className?: string }) {
  return (
    <div className={`flex justify-start ${className}`}>
      <div className="bg-gray-700/80 rounded-2xl rounded-tl-none p-4 max-w-xs shadow-md border border-gray-600/30">
        <div className="flex items-center space-x-2">
          <div className="flex space-x-1">
            <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"></div>
            <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0.2s' }}></div>
            <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0.4s' }}></div>
          </div>
          <span className="text-xs text-gray-400">IA está digitando...</span>
        </div>
      </div>
    </div>
  );
}

// Loading skeleton para lista de mensagens
export function MessageSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="space-y-4 p-4">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className={`flex ${i % 2 === 0 ? 'justify-end' : 'justify-start'}`}>
          <div className={`rounded-2xl p-4 max-w-xs md:max-w-md space-y-2 ${
            i % 2 === 0 
              ? 'bg-gray-600/30 rounded-tr-none' 
              : 'bg-gray-700/30 rounded-tl-none'
          }`}>
            <div className="h-4 bg-gray-500/30 rounded animate-pulse"></div>
            <div className="h-4 bg-gray-500/30 rounded animate-pulse w-3/4"></div>
            {Math.random() > 0.5 && (
              <div className="h-4 bg-gray-500/30 rounded animate-pulse w-1/2"></div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// Hook para controlar estados de loading
export function useLoadingStates() {
  const [loadingStates, setLoadingStates] = useState<Record<string, boolean>>({});

  const setLoading = (key: string, isLoading: boolean) => {
    setLoadingStates(prev => ({
      ...prev,
      [key]: isLoading
    }));
  };

  const isLoading = (key: string) => loadingStates[key] || false;

  const withLoading = async function<T>(key: string, asyncFn: () => Promise<T>): Promise<T> {
    setLoading(key, true);
    try {
      return await asyncFn();
    } finally {
      setLoading(key, false);
    }
  };

  return {
    loadingStates,
    setLoading,
    isLoading,
    withLoading
  };
}
