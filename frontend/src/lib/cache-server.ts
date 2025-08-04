/**
 * Cache Server-Side (Node.js)
 * Usa Redis quando disponível, fallback para memória
 */

import Redis from 'ioredis';

interface CacheEntry {
  data: any;
  timestamp: number;
  ttl: number;
  tags?: string[];
}

class ServerCacheService {
  private redis: Redis | null = null;
  private memoryCache = new Map<string, CacheEntry>();
  private isConnected = false;
  private prefix = 'ai-cache:';

  constructor() {
    this.setupRedis();
  }

  private setupRedis() {
    try {
      this.redis = new Redis({
        host: process.env.REDIS_HOST || 'localhost',
        port: parseInt(process.env.REDIS_PORT || '6379'),
        password: process.env.REDIS_PASSWORD,
        maxRetriesPerRequest: 1,
        lazyConnect: true
      });

      this.redis.on('connect', () => {
        this.isConnected = true;
        console.log('🟢 Server Cache: Redis conectado');
      });

      this.redis.on('error', (error: Error) => {
        this.isConnected = false;
        console.warn('🟡 Server Cache: Redis erro -', error.message);
      });

      this.redis.on('close', () => {
        this.isConnected = false;
        console.log('🔴 Server Cache: Redis desconectado');
      });

    } catch (error) {
      console.warn('🟡 Server Cache: Redis não disponível, usando memória');
      this.redis = null;
    }
  }

  async connect(): Promise<void> {
    if (this.redis && !this.isConnected) {
      try {
        await this.redis.connect();
      } catch (error) {
        console.warn('🟡 Server Cache: Falha ao conectar Redis:', error);
      }
    }
  }

  private formatKey(key: string): string {
    return `${this.prefix}${key}`;
  }

  async get<T>(key: string): Promise<T | null> {
    // Tentar Redis primeiro
    if (this.isConnected && this.redis) {
      try {
        const value = await this.redis.get(this.formatKey(key));
        if (value) {
          const parsed = JSON.parse(value);
          return parsed;
        }
      } catch (error) {
        console.warn('🟡 Server Cache: Erro Redis get:', error);
      }
    }

    // Fallback para memória
    const memoryEntry = this.memoryCache.get(key);
    if (memoryEntry && !this.isExpired(memoryEntry)) {
      return memoryEntry.data;
    }

    // Remover se expirado
    if (memoryEntry && this.isExpired(memoryEntry)) {
      this.memoryCache.delete(key);
    }

    return null;
  }

  async set(key: string, data: any, ttlSeconds: number = 3600, tags: string[] = []): Promise<boolean> {
    // Salvar no Redis
    if (this.isConnected && this.redis) {
      try {
        const pipeline = this.redis.pipeline();
        
        // Salvar o valor com TTL
        pipeline.setex(this.formatKey(key), ttlSeconds, JSON.stringify(data));
        
        // Adicionar tags para invalidação
        if (tags.length > 0) {
          for (const tag of tags) {
            pipeline.sadd(`${this.prefix}tag:${tag}`, key);
            pipeline.expire(`${this.prefix}tag:${tag}`, ttlSeconds);
          }
        }
        
        await pipeline.exec();
        return true;
      } catch (error) {
        console.warn('🟡 Server Cache: Erro Redis set:', error);
      }
    }

    // Fallback para memória
    const entry: CacheEntry = {
      data,
      timestamp: Date.now(),
      ttl: ttlSeconds * 1000,
      tags
    };

    this.memoryCache.set(key, entry);
    return true;
  }

  private isExpired(entry: CacheEntry): boolean {
    return Date.now() - entry.timestamp > entry.ttl;
  }

  async invalidateByTags(tags: string[]): Promise<number> {
    let removed = 0;

    // Invalidar no Redis
    if (this.isConnected && this.redis) {
      try {
        for (const tag of tags) {
          const tagKey = `${this.prefix}tag:${tag}`;
          const keys = await this.redis.smembers(tagKey);
          
          if (keys.length === 0) continue;

          const pipeline = this.redis.pipeline();
          
          // Remover todas as chaves associadas à tag
          keys.forEach((key: string) => {
            pipeline.del(this.formatKey(key));
          });
          
          // Remover o conjunto da tag
          pipeline.del(tagKey);
          
          await pipeline.exec();
          removed += keys.length;
        }
      } catch (error) {
        console.warn('🟡 Server Cache: Erro ao invalidar tags Redis:', error);
      }
    }

    // Invalidar na memória
    for (const [key, entry] of this.memoryCache) {
      if (entry.tags && entry.tags.some(tag => tags.includes(tag))) {
        this.memoryCache.delete(key);
        removed++;
      }
    }

    return removed;
  }

  getStats() {
    return {
      redisConnected: this.isConnected,
      memoryEntries: this.memoryCache.size,
      type: 'server'
    };
  }

  async flush() {
    // Limpar Redis
    if (this.isConnected && this.redis) {
      try {
        const keys = await this.redis.keys(`${this.prefix}*`);
        if (keys.length > 0) {
          await this.redis.del(...keys);
        }
      } catch (error) {
        console.warn('🟡 Server Cache: Erro ao limpar Redis:', error);
      }
    }

    // Limpar memória
    this.memoryCache.clear();
  }

  async disconnect() {
    if (this.redis) {
      await this.redis.disconnect();
    }
  }
}

/**
 * Gerenciador específico para cache de IA no servidor
 */
export class ServerAICacheManager {
  private static generateQuestionKey(question: string): string {
    // Normalizar a pergunta para melhor cache hit
    const normalized = question.toLowerCase()
      .trim()
      .replace(/[^\w\s]/g, '')
      .replace(/\s+/g, ' ');
    
    // Hash simples (não é criptográfico, apenas para cache)
    let hash = 0;
    for (let i = 0; i < normalized.length; i++) {
      const char = normalized.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    
    return `rag:q:${Math.abs(hash)}`;
  }

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

    return serverCache.set(key, response, ttlSeconds, tags);
  }

  static async getCachedRAGResponse(question: string): Promise<any | null> {
    const key = this.generateQuestionKey(question);
    return serverCache.get(key);
  }

  static async invalidateDocumentCache(filename: string) {
    return serverCache.invalidateByTags([`doc:${filename}`]);
  }
}

// Instância global para o servidor
export const serverCache = new ServerCacheService();
