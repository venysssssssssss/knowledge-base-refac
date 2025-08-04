/**
 * Sistema de notificações para feedback do usuário
 * Usado para exibir sucessos, erros e warnings de forma elegante
 */

'use client';

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface NotificationProviderProps {
  children: React.ReactNode;
}

interface NotificationContextType {
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id'>) => void;
  removeNotification: (id: string) => void;
  clearAll: () => void;
}

const NotificationContext = React.createContext<NotificationContextType | null>(null);

export function NotificationProvider({ children }: NotificationProviderProps) {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const addNotification = (notification: Omit<Notification, 'id'>) => {
    const id = Date.now().toString();
    const newNotification: Notification = {
      ...notification,
      id,
      duration: notification.duration || 5000
    };

    setNotifications(prev => [...prev, newNotification]);

    // Auto-remove após o tempo especificado
    if ((newNotification.duration ?? 0) > 0) {
      setTimeout(() => {
        removeNotification(id);
      }, newNotification.duration!);
    }
  };

  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
  };

  const clearAll = () => {
    setNotifications([]);
  };

  return (
    <NotificationContext.Provider value={{
      notifications,
      addNotification,
      removeNotification,
      clearAll
    }}>
      {children}
      <NotificationContainer 
        notifications={notifications}
        onRemove={removeNotification}
      />
    </NotificationContext.Provider>
  );
}

function NotificationContainer({ 
  notifications, 
  onRemove 
}: { 
  notifications: Notification[]; 
  onRemove: (id: string) => void;
}) {
  return (
    <div className="fixed top-4 right-4 z-50 space-y-2 max-w-sm">
      <AnimatePresence>
        {notifications.map(notification => (
          <NotificationItem
            key={notification.id}
            notification={notification}
            onRemove={onRemove}
          />
        ))}
      </AnimatePresence>
    </div>
  );
}

function NotificationItem({ 
  notification, 
  onRemove 
}: { 
  notification: Notification; 
  onRemove: (id: string) => void;
}) {
  const getStyles = (type: NotificationType) => {
    switch (type) {
      case 'success':
        return {
          bg: 'bg-green-500/10 border-green-500/30',
          icon: '✅',
          iconColor: 'text-green-400'
        };
      case 'error':
        return {
          bg: 'bg-red-500/10 border-red-500/30',
          icon: '❌',
          iconColor: 'text-red-400'
        };
      case 'warning':
        return {
          bg: 'bg-yellow-500/10 border-yellow-500/30',
          icon: '⚠️',
          iconColor: 'text-yellow-400'
        };
      case 'info':
        return {
          bg: 'bg-blue-500/10 border-blue-500/30',
          icon: 'ℹ️',
          iconColor: 'text-blue-400'
        };
    }
  };

  const styles = getStyles(notification.type);

  return (
    <motion.div
      initial={{ opacity: 0, x: 300, scale: 0.8 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 300, scale: 0.8 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={`p-4 rounded-xl border backdrop-blur-lg ${styles.bg}`}
    >
      <div className="flex items-start space-x-3">
        <span className={`text-lg ${styles.iconColor}`}>
          {styles.icon}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-white">
            {notification.title}
          </p>
          <p className="text-xs text-gray-300 mt-1">
            {notification.message}
          </p>
          {notification.action && (
            <button
              onClick={notification.action.onClick}
              className="text-xs text-emerald-400 hover:text-emerald-300 mt-2 font-medium transition-colors"
            >
              {notification.action.label}
            </button>
          )}
        </div>
        <button
          onClick={() => onRemove(notification.id)}
          className="text-gray-400 hover:text-white transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </motion.div>
  );
}

// Hook para usar o sistema de notificações
export function useNotifications() {
  const context = React.useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
}

// Utilitários para tipos específicos de notificação
export function useAINotifications() {
  const { addNotification } = useNotifications();

  return {
    notifyAISuccess: (message: string, details?: string) => {
      addNotification({
        type: 'success',
        title: 'IA Respondeu',
        message: details || message,
        duration: 4000
      });
    },

    notifyAIError: (error: string, canRetry = false) => {
      addNotification({
        type: 'error',
        title: 'Erro na IA',
        message: error,
        duration: 8000,
        action: canRetry ? {
          label: 'Tentar Novamente',
          onClick: () => window.location.reload()
        } : undefined
      });
    },

    notifyUploadSuccess: (fileName: string, chunksCreated?: number) => {
      addNotification({
        type: 'success',
        title: 'Upload Concluído',
        message: `${fileName} foi processado com sucesso${chunksCreated ? ` (${chunksCreated} fragmentos criados)` : ''}`,
        duration: 6000
      });
    },

    notifyUploadError: (fileName: string, error: string) => {
      addNotification({
        type: 'error',
        title: 'Erro no Upload',
        message: `Falha ao processar ${fileName}: ${error}`,
        duration: 8000
      });
    },

    notifyOfflineMode: () => {
      addNotification({
        type: 'warning',
        title: 'Modo Offline',
        message: 'Você está offline. Mensagens serão salvas localmente.',
        duration: 0 // Persiste até ser removida manualmente
      });
    },

    notifyConnectionRestored: () => {
      addNotification({
        type: 'info',
        title: 'Conexão Restaurada',
        message: 'Você está online novamente. Sincronizando dados...',
        duration: 3000
      });
    }
  };
}
