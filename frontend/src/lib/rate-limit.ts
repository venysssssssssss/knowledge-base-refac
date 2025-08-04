/**
 * Sistema de Rate Limiting e Throttling
 * Implementa controles de taxa para proteger os serviços de IA
 */

interface RateLimitConfig {
  windowMs: number; // Janela de tempo em ms
  maxRequests: number; // Máximo de requisições por janela
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
  keyGenerator?: (identifier: string) => string;
}

interface ThrottleConfig {
  delay: number; // Delay mínimo entre requisições em ms
  maxConcurrent: number; // Máximo de requisições simultâneas
  queueSize: number; // Tamanho máximo da fila
}

interface RateLimitEntry {
  count: number;
  resetTime: number;
  firstRequest: number;
}

interface ThrottledRequest<T = any> {
  id: string;
  execute: () => Promise<T>;
  resolve: (value: T) => void;
  reject: (error: Error) => void;
  timestamp: number;
  priority: number;
}

class RateLimiter {
  private limits = new Map<string, RateLimitEntry>();
  private config: RateLimitConfig;

  constructor(config: RateLimitConfig) {
    this.config = config;
    
    // Limpar entradas expiradas a cada minuto
    setInterval(() => {
      this.cleanup();
    }, 60000);
  }

  /**
   * Verifica se uma requisição está dentro dos limites
   */
  isAllowed(identifier: string): { allowed: boolean; resetTime?: number; remaining?: number } {
    const key = this.config.keyGenerator ? this.config.keyGenerator(identifier) : identifier;
    const now = Date.now();
    
    let entry = this.limits.get(key);
    
    if (!entry || now >= entry.resetTime) {
      // Nova janela ou primeira requisição
      entry = {
        count: 1,
        resetTime: now + this.config.windowMs,
        firstRequest: now
      };
      this.limits.set(key, entry);
      
      return {
        allowed: true,
        resetTime: entry.resetTime,
        remaining: this.config.maxRequests - 1
      };
    }
    
    if (entry.count < this.config.maxRequests) {
      entry.count++;
      return {
        allowed: true,
        resetTime: entry.resetTime,
        remaining: this.config.maxRequests - entry.count
      };
    }
    
    // Limite excedido
    return {
      allowed: false,
      resetTime: entry.resetTime,
      remaining: 0
    };
  }

  /**
   * Registra uma requisição (para estatísticas)
   */
  recordRequest(identifier: string, success: boolean) {
    if ((success && this.config.skipSuccessfulRequests) || 
        (!success && this.config.skipFailedRequests)) {
      return;
    }

    // A contagem já foi feita em isAllowed(), apenas para consistência
    this.isAllowed(identifier);
  }

  /**
   * Remove entradas expiradas
   */
  private cleanup() {
    const now = Date.now();
    const toDelete: string[] = [];
    
    for (const [key, entry] of this.limits) {
      if (now >= entry.resetTime) {
        toDelete.push(key);
      }
    }
    
    for (const key of toDelete) {
      this.limits.delete(key);
    }
    
    if (toDelete.length > 0) {
      console.log(`🧹 RateLimiter: Removidas ${toDelete.length} entradas expiradas`);
    }
  }

  /**
   * Obtém estatísticas
   */
  getStats() {
    const now = Date.now();
    const activeEntries = Array.from(this.limits.values())
      .filter(entry => now < entry.resetTime);
    
    return {
      activeEntries: activeEntries.length,
      totalEntries: this.limits.size,
      config: this.config
    };
  }
}

class RequestThrottler {
  private config: ThrottleConfig;
  private activeRequests = 0;
  private queue: ThrottledRequest<any>[] = [];
  private lastExecutionTime = 0;

  constructor(config: ThrottleConfig) {
    this.config = config;
    
    // Processar fila a cada 10ms
    setInterval(() => {
      this.processQueue();
    }, 10);
  }

  /**
   * Throttle uma função assíncrona
   */
  async throttle<R>(
    fn: () => Promise<R>,
    priority: number = 0,
    identifier?: string
  ): Promise<R> {
    return new Promise<R>((resolve, reject) => {
      const request: ThrottledRequest<R> = {
        id: identifier || `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        execute: fn as () => Promise<any>,
        resolve: resolve as (value: any) => void,
        reject,
        timestamp: Date.now(),
        priority
      };

      // Verificar se a fila está cheia
      if (this.queue.length >= this.config.queueSize) {
        reject(new Error('Throttle queue is full'));
        return;
      }

      // Adicionar à fila ordenada por prioridade
      this.queue.push(request);
      this.queue.sort((a, b) => b.priority - a.priority || a.timestamp - b.timestamp);

      console.log(`⏱️ Throttler: Adicionado à fila (${this.queue.length} pendentes)`);
    });
  }

  /**
   * Processa a fila de requisições
   */
  private async processQueue() {
    if (this.queue.length === 0 || this.activeRequests >= this.config.maxConcurrent) {
      return;
    }

    // Verificar delay mínimo
    const now = Date.now();
    if (now - this.lastExecutionTime < this.config.delay) {
      return;
    }

    const request = this.queue.shift();
    if (!request) return;

    this.activeRequests++;
    this.lastExecutionTime = now;

    console.log(`🚀 Throttler: Executando requisição ${request.id} (${this.activeRequests} ativas)`);

    try {
      const result = await request.execute();
      request.resolve(result);
    } catch (error) {
      request.reject(error instanceof Error ? error : new Error('Unknown error'));
    } finally {
      this.activeRequests--;
    }
  }

  /**
   * Obtém estatísticas
   */
  getStats() {
    return {
      activeRequests: this.activeRequests,
      queueLength: this.queue.length,
      lastExecution: this.lastExecutionTime,
      config: this.config
    };
  }

  /**
   * Limpa a fila
   */
  clearQueue() {
    const cleared = this.queue.length;
    
    // Rejeitar todas as requisições pendentes
    for (const request of this.queue) {
      request.reject(new Error('Queue cleared'));
    }
    
    this.queue = [];
    console.log(`🧹 Throttler: Fila limpa (${cleared} requisições rejeitadas)`);
    
    return cleared;
  }
}

// Rate limiters para diferentes tipos de operação
export const aiRateLimiter = new RateLimiter({
  windowMs: 60 * 1000, // 1 minuto
  maxRequests: 30, // 30 perguntas por minuto por usuário
  keyGenerator: (userId) => `ai:${userId}`
});

export const uploadRateLimiter = new RateLimiter({
  windowMs: 5 * 60 * 1000, // 5 minutos
  maxRequests: 10, // 10 uploads por 5 minutos por usuário
  keyGenerator: (userId) => `upload:${userId}`
});

export const globalRateLimiter = new RateLimiter({
  windowMs: 60 * 1000, // 1 minuto
  maxRequests: 100, // 100 requisições globais por minuto
  keyGenerator: () => 'global'
});

// Throttlers para diferentes tipos de operação
export const aiThrottler = new RequestThrottler({
  delay: 500, // 500ms entre requisições de IA
  maxConcurrent: 3, // Máximo 3 requisições de IA simultâneas
  queueSize: 50 // Fila de até 50 requisições
});

export const uploadThrottler = new RequestThrottler({
  delay: 1000, // 1s entre uploads
  maxConcurrent: 2, // Máximo 2 uploads simultâneos
  queueSize: 20 // Fila de até 20 uploads
});

/**
 * Middleware para aplicar rate limiting e throttling
 */
export class RateLimitManager {
  /**
   * Aplica rate limiting e throttling para perguntas de IA
   */
  static async checkAIRequest(userId: string): Promise<boolean> {
    // Verificar rate limits
    const userLimit = aiRateLimiter.isAllowed(userId);
    const globalLimit = globalRateLimiter.isAllowed('global');

    if (!userLimit.allowed) {
      throw new Error(`Rate limit excedido para usuário. Tente novamente em ${Math.ceil((userLimit.resetTime! - Date.now()) / 1000)}s`);
    }

    if (!globalLimit.allowed) {
      throw new Error(`Sistema sobrecarregado. Tente novamente em ${Math.ceil((globalLimit.resetTime! - Date.now()) / 1000)}s`);
    }

    return true;
  }

  /**
   * Aplica rate limiting para uploads
   */
  static async checkUploadRequest(userId: string): Promise<boolean> {
    const userLimit = uploadRateLimiter.isAllowed(userId);
    const globalLimit = globalRateLimiter.isAllowed('global');

    if (!userLimit.allowed) {
      throw new Error(`Limite de uploads excedido. Tente novamente em ${Math.ceil((userLimit.resetTime! - Date.now()) / 1000)}s`);
    }

    if (!globalLimit.allowed) {
      throw new Error(`Sistema sobrecarregado. Tente novamente em ${Math.ceil((globalLimit.resetTime! - Date.now()) / 1000)}s`);
    }

    return true;
  }

  /**
   * Throttle para requisições de IA
   */
  static async throttleAIRequest<T>(
    fn: () => Promise<T>,
    userId: string,
    priority: number = 0
  ): Promise<T> {
    await this.checkAIRequest(userId);
    
    return aiThrottler.throttle(fn, priority, `ai:${userId}`);
  }

  /**
   * Throttle para uploads
   */
  static async throttleUpload<T>(
    fn: () => Promise<T>,
    userId: string
  ): Promise<T> {
    await this.checkUploadRequest(userId);
    
    return uploadThrottler.throttle(fn, 1, `upload:${userId}`);
  }

  /**
   * Obtém estatísticas combinadas
   */
  static getStats() {
    return {
      rateLimiters: {
        ai: aiRateLimiter.getStats(),
        upload: uploadRateLimiter.getStats(),
        global: globalRateLimiter.getStats()
      },
      throttlers: {
        ai: aiThrottler.getStats(),
        upload: uploadThrottler.getStats()
      }
    };
  }
}

export type { RateLimitConfig, ThrottleConfig, RateLimitEntry, ThrottledRequest };
export { RateLimiter, RequestThrottler };
