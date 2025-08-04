/**
 * Gerenciador de Sessões de Chat
 * Implementa isolamento e paralelismo para múltiplas sessões
 */

interface ChatSession {
  id: string;
  userId: string;
  createdAt: number;
  lastActivity: number;
  activeRequests: number;
  totalRequests: number;
  status: 'active' | 'idle' | 'suspended';
  priority: number;
}

interface SessionStats {
  totalSessions: number;
  activeSessions: number;
  idleSessions: number;
  suspendedSessions: number;
  totalActiveRequests: number;
  averageResponseTime: number;
}

interface SessionConfig {
  maxConcurrentRequests: number;
  maxSessionsPerUser: number;
  sessionTimeoutMs: number;
  cleanupIntervalMs: number;
  priorityBoostThreshold: number;
}

class SessionManager {
  private sessions = new Map<string, ChatSession>();
  private userSessions = new Map<string, Set<string>>();
  private requestQueues = new Map<string, Array<() => Promise<any>>>();
  private config: SessionConfig;
  private cleanupTimer: NodeJS.Timeout | null = null;

  constructor(config?: Partial<SessionConfig>) {
    this.config = {
      maxConcurrentRequests: 3,
      maxSessionsPerUser: 5,
      sessionTimeoutMs: 30 * 60 * 1000, // 30 minutos
      cleanupIntervalMs: 5 * 60 * 1000, // 5 minutos
      priorityBoostThreshold: 10,
      ...config
    };

    this.startCleanupTimer();
  }

  /**
   * Cria ou obtém uma sessão de chat
   */
  getOrCreateSession(userId: string, sessionId?: string): ChatSession {
    const finalSessionId = sessionId || this.generateSessionId(userId);
    
    let session = this.sessions.get(finalSessionId);
    
    if (!session) {
      // Verificar limite de sessões por usuário
      const userSessionsSet = this.userSessions.get(userId) || new Set();
      
      if (userSessionsSet.size >= this.config.maxSessionsPerUser) {
        // Remover a sessão mais antiga
        const oldestSessionId = this.findOldestSession(userId);
        if (oldestSessionId) {
          this.removeSession(oldestSessionId);
        }
      }

      // Criar nova sessão
      session = {
        id: finalSessionId,
        userId,
        createdAt: Date.now(),
        lastActivity: Date.now(),
        activeRequests: 0,
        totalRequests: 0,
        status: 'active',
        priority: 0
      };

      this.sessions.set(finalSessionId, session);
      
      // Adicionar às sessões do usuário
      if (!this.userSessions.has(userId)) {
        this.userSessions.set(userId, new Set());
      }
      this.userSessions.get(userId)!.add(finalSessionId);

      // Inicializar fila de requisições
      this.requestQueues.set(finalSessionId, []);

      console.log(`📱 Nova sessão criada: ${finalSessionId} para usuário ${userId}`);
    } else {
      // Atualizar atividade da sessão existente
      session.lastActivity = Date.now();
      session.status = 'active';
    }

    return session;
  }

  /**
   * Executa uma requisição dentro do contexto de uma sessão
   */
  async executeInSession<T>(
    sessionId: string,
    requestFn: () => Promise<T>,
    priority: number = 0
  ): Promise<T> {
    const session = this.sessions.get(sessionId);
    
    if (!session) {
      throw new Error(`Sessão ${sessionId} não encontrada`);
    }

    // Verificar se a sessão pode aceitar mais requisições
    if (session.activeRequests >= this.config.maxConcurrentRequests) {
      // Adicionar à fila
      return this.queueRequest(sessionId, requestFn, priority);
    }

    return this.executeRequest(sessionId, requestFn);
  }

  /**
   * Adiciona uma requisição à fila da sessão
   */
  private async queueRequest<T>(
    sessionId: string,
    requestFn: () => Promise<T>,
    priority: number
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const queue = this.requestQueues.get(sessionId) || [];
      
      const queuedRequest = async () => {
        try {
          const result = await this.executeRequest(sessionId, requestFn);
          resolve(result);
        } catch (error) {
          reject(error);
        }
      };

      // Inserir na posição correta baseado na prioridade
      const insertIndex = queue.findIndex((_, index) => {
        // Prioridade mais alta = executar primeiro
        return priority > (queue[index] as any).priority || 0;
      });

      if (insertIndex === -1) {
        queue.push(queuedRequest);
      } else {
        queue.splice(insertIndex, 0, queuedRequest);
      }

      (queuedRequest as any).priority = priority;
      this.requestQueues.set(sessionId, queue);

      console.log(`⏳ Requisição adicionada à fila da sessão ${sessionId} (${queue.length} na fila)`);
    });
  }

  /**
   * Executa uma requisição e gerencia o estado da sessão
   */
  private async executeRequest<T>(
    sessionId: string,
    requestFn: () => Promise<T>
  ): Promise<T> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Sessão ${sessionId} não encontrada`);
    }

    session.activeRequests++;
    session.totalRequests++;
    session.lastActivity = Date.now();

    const startTime = Date.now();

    try {
      console.log(`🚀 Executando requisição na sessão ${sessionId} (${session.activeRequests} ativas)`);
      
      const result = await requestFn();
      
      const duration = Date.now() - startTime;
      console.log(`✅ Requisição completada na sessão ${sessionId} em ${duration}ms`);
      
      // Aumentar prioridade se sessão está sendo muito usada
      if (session.totalRequests > this.config.priorityBoostThreshold) {
        session.priority = Math.min(session.priority + 1, 10);
      }

      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      console.log(`❌ Requisição falhou na sessão ${sessionId} em ${duration}ms:`, error);
      throw error;
    } finally {
      session.activeRequests--;
      
      // Processar próxima requisição na fila
      this.processNextInQueue(sessionId);
    }
  }

  /**
   * Processa a próxima requisição na fila da sessão
   */
  private processNextInQueue(sessionId: string) {
    const session = this.sessions.get(sessionId);
    const queue = this.requestQueues.get(sessionId);
    
    if (!session || !queue || queue.length === 0) {
      return;
    }

    if (session.activeRequests < this.config.maxConcurrentRequests) {
      const nextRequest = queue.shift();
      if (nextRequest) {
        // Executar a próxima requisição da fila
        nextRequest();
      }
    }
  }

  /**
   * Remove uma sessão e limpa recursos
   */
  removeSession(sessionId: string): boolean {
    const session = this.sessions.get(sessionId);
    
    if (!session) {
      return false;
    }

    // Remover das sessões do usuário
    const userSessionsSet = this.userSessions.get(session.userId);
    if (userSessionsSet) {
      userSessionsSet.delete(sessionId);
      if (userSessionsSet.size === 0) {
        this.userSessions.delete(session.userId);
      }
    }

    // Cancelar requisições pendentes
    const queue = this.requestQueues.get(sessionId);
    if (queue && queue.length > 0) {
      console.log(`🗑️ Cancelando ${queue.length} requisições pendentes da sessão ${sessionId}`);
    }

    this.sessions.delete(sessionId);
    this.requestQueues.delete(sessionId);

    console.log(`🗑️ Sessão removida: ${sessionId}`);
    return true;
  }

  /**
   * Encontra a sessão mais antiga de um usuário
   */
  private findOldestSession(userId: string): string | null {
    const userSessionsSet = this.userSessions.get(userId);
    
    if (!userSessionsSet || userSessionsSet.size === 0) {
      return null;
    }

    let oldestSession: ChatSession | null = null;
    let oldestSessionId: string | null = null;

    for (const sessionId of userSessionsSet) {
      const session = this.sessions.get(sessionId);
      if (session && (!oldestSession || session.lastActivity < oldestSession.lastActivity)) {
        oldestSession = session;
        oldestSessionId = sessionId;
      }
    }

    return oldestSessionId;
  }

  /**
   * Gera um ID único para a sessão
   */
  private generateSessionId(userId: string): string {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substr(2, 5);
    return `chat_${userId}_${timestamp}_${random}`;
  }

  /**
   * Inicia o timer de limpeza automática
   */
  private startCleanupTimer() {
    this.cleanupTimer = setInterval(() => {
      this.cleanupExpiredSessions();
    }, this.config.cleanupIntervalMs);
  }

  /**
   * Remove sessões expiradas
   */
  private cleanupExpiredSessions() {
    const now = Date.now();
    const expiredSessions: string[] = [];

    for (const [sessionId, session] of this.sessions) {
      const timeSinceLastActivity = now - session.lastActivity;
      
      if (timeSinceLastActivity > this.config.sessionTimeoutMs && session.activeRequests === 0) {
        expiredSessions.push(sessionId);
      } else if (timeSinceLastActivity > this.config.sessionTimeoutMs / 2) {
        // Marcar como idle se mais da metade do timeout passou
        session.status = 'idle';
      }
    }

    if (expiredSessions.length > 0) {
      console.log(`🧹 Limpando ${expiredSessions.length} sessões expiradas`);
      
      for (const sessionId of expiredSessions) {
        this.removeSession(sessionId);
      }
    }
  }

  /**
   * Obtém estatísticas das sessões
   */
  getStats(): SessionStats {
    const sessions = Array.from(this.sessions.values());
    
    return {
      totalSessions: sessions.length,
      activeSessions: sessions.filter(s => s.status === 'active').length,
      idleSessions: sessions.filter(s => s.status === 'idle').length,
      suspendedSessions: sessions.filter(s => s.status === 'suspended').length,
      totalActiveRequests: sessions.reduce((sum, s) => sum + s.activeRequests, 0),
      averageResponseTime: 0 // TODO: implementar tracking de response time
    };
  }

  /**
   * Lista sessões de um usuário
   */
  getUserSessions(userId: string): ChatSession[] {
    const userSessionsSet = this.userSessions.get(userId);
    
    if (!userSessionsSet) {
      return [];
    }

    return Array.from(userSessionsSet)
      .map(sessionId => this.sessions.get(sessionId))
      .filter((session): session is ChatSession => session !== undefined)
      .sort((a, b) => b.lastActivity - a.lastActivity);
  }

  /**
   * Para o gerenciador e limpa recursos
   */
  destroy() {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = null;
    }

    this.sessions.clear();
    this.userSessions.clear();
    this.requestQueues.clear();
  }
}

// Instância global do gerenciador de sessões
export const sessionManager = new SessionManager();

export type { ChatSession, SessionStats, SessionConfig };
export { SessionManager };
