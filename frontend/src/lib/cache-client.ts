/**
 * Cache Client-Side (Browser)
 * Usa IndexedDB e localStorage como fallback
 */

interface CacheEntry {
  data: any;
  timestamp: number;
  ttl: number;
  tags?: string[];
}

class BrowserCacheService {
  private memoryCache = new Map<string, CacheEntry>();
  private dbName = 'ai-cache';
  private storeName = 'cache-entries';
  private db: IDBDatabase | null = null;
  private connected = false;

  async connect(): Promise<void> {
    if (this.connected) return;

    try {
      this.db = await this.openDB();
      this.connected = true;
      console.log('🟢 Browser Cache: IndexedDB conectado');
    } catch (error) {
      console.warn('🟡 Browser Cache: Usando apenas memória', error);
      this.connected = false;
    }
  }

  private openDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, { keyPath: 'key' });
          store.createIndex('timestamp', 'timestamp');
          store.createIndex('tags', 'tags', { multiEntry: true });
        }
      };
    });
  }

  async get<T>(key: string): Promise<T | null> {
    // Tentar cache em memória primeiro
    const memoryEntry = this.memoryCache.get(key);
    if (memoryEntry && !this.isExpired(memoryEntry)) {
      return memoryEntry.data;
    }

    // Remover se expirado
    if (memoryEntry && this.isExpired(memoryEntry)) {
      this.memoryCache.delete(key);
    }

    // Tentar IndexedDB
    if (this.connected && this.db) {
      try {
        const entry = await this.getFromDB(key);
        if (entry && !this.isExpired(entry)) {
          // Colocar de volta na memória
          this.memoryCache.set(key, entry);
          return entry.data;
        }
      } catch (error) {
        console.warn('🟡 Browser Cache: Erro no IndexedDB:', error);
      }
    }

    return null;
  }

  async set(key: string, data: any, ttlSeconds: number = 3600, tags: string[] = []): Promise<boolean> {
    const entry: CacheEntry = {
      data,
      timestamp: Date.now(),
      ttl: ttlSeconds * 1000,
      tags
    };

    // Salvar na memória
    this.memoryCache.set(key, entry);

    // Salvar no IndexedDB
    if (this.connected && this.db) {
      try {
        await this.saveToDB(key, entry);
      } catch (error) {
        console.warn('🟡 Browser Cache: Erro ao salvar no IndexedDB:', error);
      }
    }

    return true;
  }

  private getFromDB(key: string): Promise<CacheEntry | null> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('DB not connected'));
        return;
      }

      const transaction = this.db.transaction([this.storeName], 'readonly');
      const store = transaction.objectStore(this.storeName);
      const request = store.get(key);

      request.onsuccess = () => {
        const result = request.result;
        resolve(result ? { ...result, key: undefined } : null);
      };

      request.onerror = () => reject(request.error);
    });
  }

  private saveToDB(key: string, entry: CacheEntry): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('DB not connected'));
        return;
      }

      const transaction = this.db.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);
      const request = store.put({ key, ...entry });

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  private isExpired(entry: CacheEntry): boolean {
    return Date.now() - entry.timestamp > entry.ttl;
  }

  async invalidateByTags(tags: string[]): Promise<number> {
    let removed = 0;

    // Remover da memória
    for (const [key, entry] of this.memoryCache) {
      if (entry.tags && entry.tags.some(tag => tags.includes(tag))) {
        this.memoryCache.delete(key);
        removed++;
      }
    }

    // Remover do IndexedDB
    if (this.connected && this.db) {
      try {
        const transaction = this.db.transaction([this.storeName], 'readwrite');
        const store = transaction.objectStore(this.storeName);
        const index = store.index('tags');

        for (const tag of tags) {
          const request = index.openCursor(IDBKeyRange.only(tag));
          request.onsuccess = (event) => {
            const cursor = (event.target as IDBRequest).result;
            if (cursor) {
              cursor.delete();
              cursor.continue();
            }
          };
        }
      } catch (error) {
        console.warn('🟡 Browser Cache: Erro ao invalidar tags:', error);
      }
    }

    return removed;
  }

  getStats() {
    return {
      memoryEntries: this.memoryCache.size,
      connected: this.connected,
      type: 'browser'
    };
  }

  flush() {
    this.memoryCache.clear();
    
    if (this.connected && this.db) {
      try {
        const transaction = this.db.transaction([this.storeName], 'readwrite');
        const store = transaction.objectStore(this.storeName);
        store.clear();
      } catch (error) {
        console.warn('🟡 Browser Cache: Erro ao limpar IndexedDB:', error);
      }
    }
  }
}

/**
 * Gerenciador específico para cache de IA no browser
 */
export class BrowserAICacheManager {
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

    return browserCache.set(key, response, ttlSeconds, tags);
  }

  static async getCachedRAGResponse(question: string): Promise<any | null> {
    const key = this.generateQuestionKey(question);
    return browserCache.get(key);
  }

  static async invalidateDocumentCache(filename: string) {
    return browserCache.invalidateByTags([`doc:${filename}`]);
  }
}

// Instância global para o browser
export const browserCache = new BrowserCacheService();
