/**
 * Cache Universal - Detecta ambiente e usa a implementação apropriada
 */

import { browserCache, BrowserAICacheManager } from './cache-client';

// Detecta se está no browser
const isBrowser = typeof window !== 'undefined';

// Cache manager universal
export class UniversalCacheManager {
  static async getCachedRAGResponse(question: string): Promise<any | null> {
    if (isBrowser) {
      return BrowserAICacheManager.getCachedRAGResponse(question);
    } else {
      // No servidor, usar via API
      return null; // Deixar para o servidor handle
    }
  }

  static async cacheRAGResponse(
    question: string,
    response: any,
    ttlSeconds: number = 1800
  ): Promise<boolean> {
    if (isBrowser) {
      return BrowserAICacheManager.cacheRAGResponse(question, response, ttlSeconds);
    } else {
      // No servidor, usar via API
      return true; // Deixar para o servidor handle
    }
  }

  static async invalidateDocumentCache(filename: string): Promise<number> {
    if (isBrowser) {
      return BrowserAICacheManager.invalidateDocumentCache(filename);
    } else {
      return 0; // Deixar para o servidor handle
    }
  }

  static async getStats() {
    if (isBrowser) {
      return browserCache.getStats();
    } else {
      return { type: 'server-proxy' };
    }
  }

  static async flush() {
    if (isBrowser) {
      browserCache.flush();
    }
  }
}

// Cache service universal
export class UniversalCacheService {
  static async connect(): Promise<void> {
    if (isBrowser) {
      await browserCache.connect();
    }
    // No servidor, a conexão é feita automaticamente
  }

  static async get<T>(key: string): Promise<T | null> {
    if (isBrowser) {
      return browserCache.get<T>(key);
    } else {
      return null; // Deixar para o servidor handle
    }
  }

  static async set(key: string, data: any, ttlSeconds: number = 3600, tags: string[] = []): Promise<boolean> {
    if (isBrowser) {
      return browserCache.set(key, data, ttlSeconds, tags);
    } else {
      return true; // Deixar para o servidor handle
    }
  }

  static async invalidateByTags(tags: string[]): Promise<number> {
    if (isBrowser) {
      return browserCache.invalidateByTags(tags);
    } else {
      return 0; // Deixar para o servidor handle
    }
  }

  static getStats() {
    if (isBrowser) {
      return browserCache.getStats();
    } else {
      return { type: 'server-proxy' };
    }
  }

  static flush() {
    if (isBrowser) {
      browserCache.flush();
    }
  }
}

// Exports para compatibilidade
export const cacheService = UniversalCacheService;
export const AICacheManager = UniversalCacheManager;
