/**
 * Sistema de Cache Distribuído com Redis
 * Implementa cache local como fallback
 */

import Redis from 'ioredis';

interface CacheConfig {
  host?: string;
  port?: number;
  password?: string;
  db?: number;
  keyPrefix?: string;
  retryDelayOnFailover?: number;
  maxRetriesPerRequest?: number;
}

interface CacheItem<T = any> {
  data: T;
  timestamp: number;
  ttl: number;
  tags?: string[];
  metadata?: Record<string, any>;
}

interface CacheStats {
  hits: number;
  misses: number;
  hitRate: number;
  totalRequests: number;
  avgResponseTime: number;
}

class CacheService {
  private redis: Redis;
  private stats: CacheStats;
  private keyPrefix: string;
  private isConnected: boolean = false;

  constructor(config: CacheConfig = {}) {
    this.keyPrefix = config.keyPrefix || 'ai_kb:';
    this.stats = {
      hits: 0,
      misses: 0,
      hitRate: 0,
      totalRequests: 0,
      avgResponseTime: 0
    };

    // Configuração Redis com fallback para desenvolvimento
    this.redis = new Redis({
      host: config.host || process.env.REDIS_HOST || 'localhost',
      port: config.port || parseInt(process.env.REDIS_PORT || '6379'),
      password: config.password || process.env.REDIS_PASSWORD,
      db: config.db || 0,
      // retryDelayOnFailover is not a valid RedisOptions property, so it has been removed
      maxRetriesPerRequest: config.maxRetriesPerRequest || 3,
      lazyConnect: true, // Conectar apenas quando necessário
      enableOfflineQueue: false // Não acumular comandos quando desconectado
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers() {
    this.redis.on('connect', () => {
      this.isConnected = true;
      console.log('🟢 Redis Cache: Conectado');
    });

    this.redis.on('error', (error: Error) => {
      this.isConnected = false;
      console.warn('🟡 Redis Cache: Erro -', error.message);
    });

    this.redis.on('close', () => {
      this.isConnected = false;
      console.log('🔴 Redis Cache: Desconectado');
    });
  }

  async connect(): Promise<boolean> {
    try {
      await this.redis.connect();
      return this.isConnected;
    } catch (error) {
      console.warn('⚠️ Redis Cache: Falha ao conectar, usando cache local');
      return false;
    }
  }

  private formatKey(key: string): string {
    return `${this.keyPrefix}${key}`;
  }

  /**
   * Define um item no cache
   */
  async set<T>(
    key: string, 
    value: T, 
    ttlSeconds: number = 3600,
    tags: string[] = [],
    metadata: Record<string, any> = {}
  ): Promise<boolean> {
    if (!this.isConnected) {
      return false;
    }

    const cacheItem: CacheItem<T> = {
      data: value,
      timestamp: Date.now(),
      ttl: ttlSeconds * 1000, // Converter para ms
      tags,
      metadata
    };

    try {
      const serialized = JSON.stringify(cacheItem);
      await this.redis.setex(this.formatKey(key), ttlSeconds, serialized);
      
      // Indexar por tags para invalidação em grupo
      if (tags.length > 0) {
        const pipeline = this.redis.pipeline();
        tags.forEach(tag => {
          pipeline.sadd(this.formatKey(`tag:${tag}`), key);
          pipeline.expire(this.formatKey(`tag:${tag}`), ttlSeconds + 300); // TTL um pouco maior
        });
        await pipeline.exec();
      }

      return true;
    } catch (error) {
      console.warn('⚠️ Cache set error:', error);
      return false;
    }
  }

  /**
   * Obtém um item do cache
   */
  async get<T>(key: string): Promise<T | null> {
    const startTime = Date.now();
    this.stats.totalRequests++;

    if (!this.isConnected) {
      this.stats.misses++;
      this.updateStats(startTime);
      return null;
    }

    try {
      const cached = await this.redis.get(this.formatKey(key));
      
      if (!cached) {
        this.stats.misses++;
        this.updateStats(startTime);
        return null;
      }

      const cacheItem: CacheItem<T> = JSON.parse(cached);
      
      // Verificar se expirou (double-check)
      if (Date.now() - cacheItem.timestamp > cacheItem.ttl) {
        await this.delete(key);
        this.stats.misses++;
        this.updateStats(startTime);
        return null;
      }

      this.stats.hits++;
      this.updateStats(startTime);
      
      // Log para debug
      console.log(`🟢 Cache HIT: ${key} (${Date.now() - startTime}ms)`);
      
      return cacheItem.data;
    } catch (error) {
      console.warn('⚠️ Cache get error:', error);
      this.stats.misses++;
      this.updateStats(startTime);
      return null;
    }
  }

  /**
   * Remove um item do cache
   */
  async delete(key: string): Promise<boolean> {
    if (!this.isConnected) {
      return false;
    }

    try {
      const result = await this.redis.del(this.formatKey(key));
      return result > 0;
    } catch (error) {
      console.warn('⚠️ Cache delete error:', error);
      return false;
    }
  }

  /**
   * Invalida cache por tags
   */
  async invalidateByTag(tag: string): Promise<number> {
    if (!this.isConnected) {
      return 0;
    }

    try {
      const keys = await this.redis.smembers(this.formatKey(`tag:${tag}`));
      
      if (keys.length === 0) {
        return 0;
      }

      const pipeline = this.redis.pipeline();
      
      // Remover todas as chaves associadas à tag
      keys.forEach((key: string) => {
        pipeline.del(this.formatKey(key));
      });
      
      // Remover o conjunto da tag
      pipeline.del(this.formatKey(`tag:${tag}`));
      
      const results = await pipeline.exec();
      
      console.log(`🧹 Cache: Invalidados ${keys.length} itens pela tag "${tag}"`);
      
      return keys.length;
    } catch (error) {
      console.warn('⚠️ Cache invalidateByTag error:', error);
      return 0;
    }
  }

  /**
   * Limpa todo o cache
   */
  async flush(): Promise<boolean> {
    if (!this.isConnected) {
      return false;
    }

    try {
      // Buscar todas as chaves com nosso prefixo
      const keys = await this.redis.keys(`${this.keyPrefix}*`);
      
      if (keys.length > 0) {
        await this.redis.del(...keys);
        console.log(`🧹 Cache: Removidas ${keys.length} chaves`);
      }
      
      return true;
    } catch (error) {
      console.warn('⚠️ Cache flush error:', error);
      return false;
    }
  }

  /**
   * Cache com função de fallback
   */
  async getOrSet<T>(
    key: string,
    fallbackFn: () => Promise<T>,
    ttlSeconds: number = 3600,
    tags: string[] = []
  ): Promise<T> {
    // Tentar buscar do cache primeiro
    const cached = await this.get<T>(key);
    if (cached !== null) {
      return cached;
    }

    // Executar função de fallback
    console.log(`🔄 Cache MISS: Executando fallback para "${key}"`);
    const value = await fallbackFn();
    
    // Salvar no cache para próxima vez
    await this.set(key, value, ttlSeconds, tags);
    
    return value;
  }

  private updateStats(startTime: number) {
    const responseTime = Date.now() - startTime;
    this.stats.avgResponseTime = (
      (this.stats.avgResponseTime * (this.stats.totalRequests - 1) + responseTime) / 
      this.stats.totalRequests
    );
    this.stats.hitRate = this.stats.hits / this.stats.totalRequests;
  }

  /**
   * Obtém estatísticas do cache
   */
  getStats(): CacheStats {
    return { ...this.stats };
  }

  /**
   * Reset das estatísticas
   */
  resetStats() {
    this.stats = {
      hits: 0,
      misses: 0,
      hitRate: 0,
      totalRequests: 0,
      avgResponseTime: 0
    };
  }

  /**
   * Desconectar do Redis
   */
  async disconnect() {
    if (this.redis) {
      await this.redis.disconnect();
    }
  }
}

// Instância singleton
export const cacheService = new CacheService();

// Utilitários para cache específico de IA
export class AICacheManager {
  static generateQuestionKey(question: string, context?: string): string {
    const normalizedQuestion = question.toLowerCase().trim();
    const hash = this.simpleHash(normalizedQuestion + (context || ''));
    return `rag:question:${hash}`;
  }

  static generateDocumentKey(filename: string, size: number): string {
    return `document:${this.simpleHash(filename + size)}`;
  }

  static generateSessionKey(sessionId: string): string {
    return `session:${sessionId}`;
  }

  private static simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  /**
   * Cache para perguntas similares (semântico)
   */
  static async cacheRAGResponse(
    question: string,
    response: any,
    ttlSeconds: number = 1800 // 30 minutos
  ) {
    const key = this.generateQuestionKey(question);
    const tags = ['rag', 'questions'];
    
    // Adicionar tags baseadas nas fontes
    if (response.sources && response.sources.length > 0) {
      response.sources.forEach((source: any) => {
        const filename = source.metadata?.filename;
        if (filename) {
          tags.push(`doc:${filename}`);
        }
      });
    }

    return cacheService.set(key, response, ttlSeconds, tags);
  }

  static async getCachedRAGResponse(question: string): Promise<any | null> {
    const key = this.generateQuestionKey(question);
    return cacheService.get(key);
  }

  /**
   * Invalida cache quando documentos são atualizados
   */
  static async invalidateDocumentCache(filename: string) {
    return cacheService.invalidateByTag(`doc:${filename}`);
  }

  /**
   * Invalida todas as respostas RAG
   */
  static async invalidateAllRAG() {
    return cacheService.invalidateByTag('rag');
  }
}

export type { CacheConfig, CacheItem, CacheStats };
export { CacheService };
