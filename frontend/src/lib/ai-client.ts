/**
 * Cliente HTTP para comunicação com AI Services
 * Implementa padrões de resiliência e cache distribuído
 */

import { cacheService as universalCache, AICacheManager } from './cache-universal';
import { SmartQueue } from './queue-service';
import { RateLimitManager } from './rate-limit';
import { sessionManager } from './session-manager';

// Instâncias globais dos serviços
const smartQueue = new SmartQueue();

interface AIServiceConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
  retryDelay: number;
}

interface RAGRequest {
  question: string;
  max_tokens?: number;
  temperature?: number;
  search_limit?: number;
  score_threshold?: number;
  document_id?: string;
}

interface RAGResponse {
  question: string;
  answer: string;
  sources: Array<{
    content: string;
    metadata: {
      chunk_index: any;
      filename: string;
      page?: number;
    };
    score: number;
  }>;
  tokens_used: number;
  processing_time: number;
  search_time: number;
  generation_time: number;
}

interface UploadResponse {
  success: boolean;
  message: string;
  document_id?: string;
  chunks_created?: number;
  metadata?: {
    filename: string;
    size: number;
    type: string;
    processed_at: string;
  };
}

class AIServiceError extends Error {
  constructor(
    message: string, 
    public status?: number, 
    public service?: string
  ) {
    super(message);
    this.name = 'AIServiceError';
  }
}

class AIClient {
  private config: AIServiceConfig;
  private cache = new Map<string, { data: any; timestamp: number; ttl: number }>();
  private initialized = false;

  constructor(config?: Partial<AIServiceConfig>) {
    this.config = {
      baseUrl: (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:3000') + '/api/ai',
      timeout: 30000,
      retries: 3,
      retryDelay: 1000,
      ...config
    };
  }

  private async ensureCacheInitialized() {
    if (!this.initialized) {
      try {
        await universalCache.connect();
        
        // Registrar processadores de fila
        this.setupQueueProcessors();
        
        this.initialized = true;
        console.log('🟢 AI Client: Cache distribuído e filas conectados');
      } catch (error) {
        console.warn('🟡 AI Client: Usando cache local como fallback');
        this.setupQueueProcessors(); // Ainda configurar filas
        this.initialized = true;
      }
    }
  }

  private setupQueueProcessors() {
    // Processador para perguntas RAG individuais
    smartQueue.registerProcessor('rag-question', async (job) => {
      const request = job.data as RAGRequest;
      console.log(`🤖 Processando pergunta RAG: "${request.question.substring(0, 50)}..."`);
      
      return this.makeDirectRequest('/rag/query', {
        method: 'POST',
        body: JSON.stringify(request),
      });
    });

    // Processador em lote para perguntas similares
    smartQueue.registerBatchProcessor('rag-question', async (jobs) => {
      console.log(`📦 Processando lote de ${jobs.length} perguntas similares`);
      
      const results = [];
      for (const job of jobs) {
        try {
          const result = await this.makeDirectRequest('/rag/query', {
            method: 'POST',
            body: JSON.stringify(job.data),
          });
          results.push(result);
        } catch (error) {
          results.push({ error: error instanceof Error ? error.message : 'Unknown error' });
        }
      }
      
      return results;
    });

    // Processador para uploads
    smartQueue.registerProcessor('document-upload', async (job) => {
      const file = job.data as File;
      console.log(`📄 Processando upload: ${file.name}`);
      
      const formData = new FormData();
      formData.append('file', file);
      
      return this.makeDirectRequest('/document/upload', {
        method: 'POST',
        body: formData,
        headers: {}, // Remover content-type para FormData
      });
    });
  }

  /**
   * Faz requisição direta sem cache (usado pelos processadores de fila)
   */
  private async makeDirectRequest(endpoint: string, options: RequestInit): Promise<any> {
    const url = `${this.config.baseUrl}${endpoint}`;
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= this.config.retries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

        const response = await fetch(url, {
          ...options,
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json',
            ...options.headers,
          },
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.text();
          throw new AIServiceError(
            `AI Service error: ${response.status} - ${errorText}`,
            response.status,
            endpoint.split('/')[1]
          );
        }

        return await response.json();

      } catch (error) {
        lastError = error as Error;
        if (attempt < this.config.retries) {
          await new Promise(resolve => 
            setTimeout(resolve, this.config.retryDelay * attempt)
          );
        }
      }
    }

    throw lastError || new AIServiceError('Unknown error occurred');
  }

  private async makeRequest<T>(
    endpoint: string, 
    options: RequestInit = {}, 
    cacheKey?: string,
    cacheTTL: number = 60000 // 1 minuto default
  ): Promise<T> {
    await this.ensureCacheInitialized();

    // Verificar cache distribuído primeiro
    if (cacheKey) {
      const cached = await universalCache.get<T>(cacheKey);
      if (cached !== null) {
        console.log(`🟢 Cache distribuído HIT: ${cacheKey}`);
        return cached;
      }

      // Fallback para cache local
      if (this.cache.has(cacheKey)) {
        const cached = this.cache.get(cacheKey)!;
        if (Date.now() - cached.timestamp < cached.ttl) {
          console.log(`� Cache local HIT: ${cacheKey}`);
          return cached.data;
        }
        this.cache.delete(cacheKey);
      }
    }

    const url = `${this.config.baseUrl}${endpoint}`;
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= this.config.retries; attempt++) {
      try {
        console.log(`🚀 AI Service Request [Attempt ${attempt}]: ${endpoint}`);
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

        const response = await fetch(url, {
          ...options,
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json',
            ...options.headers,
          },
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.text();
          throw new AIServiceError(
            `AI Service error: ${response.status} - ${errorText}`,
            response.status,
            endpoint.split('/')[1]
          );
        }

        const data = await response.json();
        
        // Armazenar no cache distribuído primeiro, fallback para local
        if (cacheKey) {
          const cacheSuccess = await universalCache.set(cacheKey, data, Math.floor(cacheTTL / 1000));
          
          if (!cacheSuccess) {
            // Fallback para cache local
            this.cache.set(cacheKey, {
              data,
              timestamp: Date.now(),
              ttl: cacheTTL
            });
            console.log(`💾 Cache local: ${cacheKey}`);
          } else {
            console.log(`💾 Cache distribuído: ${cacheKey}`);
          }
        }

        console.log(`✅ AI Service Success: ${endpoint}`);
        return data;

      } catch (error) {
        lastError = error as Error;
        console.warn(`⚠️ AI Service attempt ${attempt} failed:`, error);

        if (attempt < this.config.retries) {
          await new Promise(resolve => 
            setTimeout(resolve, this.config.retryDelay * attempt)
          );
        }
      }
    }

    console.error(`❌ AI Service failed after ${this.config.retries} attempts:`, lastError);
    throw lastError || new AIServiceError('Unknown error occurred');
  }

  /**
   * Envia uma pergunta para o RAG com rate limiting e cache
   */
  async askQuestion(
    request: RAGRequest, 
    userId: string = 'anonymous', 
    sessionId?: string
  ): Promise<RAGResponse> {
    await this.ensureCacheInitialized();

    console.log('🤖 AI Client: Enviando pergunta para RAG:', request.question);

    // Gerenciar sessão
    const session = sessionManager.getOrCreateSession(userId, sessionId);
    
    return sessionManager.executeInSession(session.id, async () => {
      // Rate limiting
      try {
      await RateLimitManager.throttleAIRequest(async () => {
        return Promise.resolve(); // Apenas verificação de rate limit
      }, userId, 1);
    } catch (error) {
      throw new AIServiceError(error instanceof Error ? error.message : 'Rate limit excedido');
    }

    // Verificar cache primeiro
    const cachedResponse = await AICacheManager.getCachedRAGResponse(request.question);
    if (cachedResponse) {
      console.log('� AI Client: Resposta obtida do cache');
      return cachedResponse as RAGResponse;
    }

    // Adicionar à fila com throttling automático
    const jobId = await smartQueue.addJob('rag-question', request, {
      priority: 1
    });

    const response = await smartQueue.waitForJob(jobId, 60000); // 60s timeout

      // Cache da resposta
      if (response && response.answer) {
        await AICacheManager.cacheRAGResponse(
          request.question,
          response,
          1800 // 30 minutos TTL
        );
      }

      return response as RAGResponse;
    }, 1); // Prioridade normal para perguntas
  }

  /**
   * Faz upload e processamento de documento
   * Usa sistema de filas para otimizar processamento
   */
  async uploadDocument(file: File): Promise<UploadResponse> {
    await this.ensureCacheInitialized();

    console.log('📤 Adicionando upload à fila de processamento...');
    const jobId = await smartQueue.addJob('document-upload', file, {
      priority: 2 // Prioridade alta para uploads
    });

    // Aguardar processamento
    const response = await smartQueue.waitForJob(jobId, 120000); // 2 minutos timeout para uploads

    return response;
  }

  /**
   * Busca documentos por query
   */
  async searchDocuments(query: string, limit: number = 5) {
    const cacheKey = `search:${query}:${limit}`;
    
    return this.makeRequest(
      '/search',
      {
        method: 'POST',
        body: JSON.stringify({ query, limit }),
      },
      cacheKey,
      180000 // 3 minutos para buscas
    );
  }

  /**
   * Verifica saúde dos serviços
   */
  async healthCheck() {
    return this.makeRequest('/health', { method: 'GET' });
  }

  /**
   * Obtém estatísticas do cache
   */
  getCacheStats() {
    return universalCache.getStats();
  }

  /**
   * Obtém estatísticas da fila
   */
  getQueueStats() {
    return smartQueue.getStats();
  }

  /**
   * Limpa cache manualmente
   */
  clearCache() {
    this.cache.clear();
    universalCache.flush();
    console.log('🧹 AI Client cache cleared');
  }

  /**
   * Obtém estatísticas das sessões
   */
  getSessionStats() {
    return sessionManager.getStats();
  }

  /**
   * Lista sessões de um usuário
   */
  getUserSessions(userId: string) {
    return sessionManager.getUserSessions(userId);
  }

  /**
   * Remove uma sessão específica
   */
  removeSession(sessionId: string) {
    return sessionManager.removeSession(sessionId);
  }

  /**
   * Obtém estatísticas combinadas
   */
  getStats() {
    const now = Date.now();
    const valid = Array.from(this.cache.values()).filter(
      item => now - item.timestamp < item.ttl
    ).length;
    
    return {
      localCache: {
        total: this.cache.size,
        valid,
        expired: this.cache.size - valid
      },
      distributedCache: this.getCacheStats(),
      queue: this.getQueueStats(),
      sessions: this.getSessionStats()
    };
  }
}

// Instância singleton
export const aiClient = new AIClient();

// Tipos exportados
export type { RAGRequest, RAGResponse, UploadResponse, AIServiceConfig };
export { AIServiceError };
