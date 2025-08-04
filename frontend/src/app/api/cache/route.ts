/**
 * API Route para Cache no Servidor
 * Gerencia cache Redis/memória no backend
 */

import { NextRequest, NextResponse } from 'next/server';
import { serverCache, ServerAICacheManager } from '@/lib/cache-server';

// Inicializar cache do servidor
await serverCache.connect();

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action');
    const key = searchParams.get('key');

    switch (action) {
      case 'get':
        if (!key) {
          return NextResponse.json({ error: 'Key required' }, { status: 400 });
        }
        const value = await serverCache.get(key);
        return NextResponse.json({ value });

      case 'get-rag':
        const question = searchParams.get('question');
        if (!question) {
          return NextResponse.json({ error: 'Question required' }, { status: 400 });
        }
        const ragResponse = await ServerAICacheManager.getCachedRAGResponse(question);
        return NextResponse.json({ value: ragResponse });

      case 'stats':
        const stats = serverCache.getStats();
        return NextResponse.json({ stats });

      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
    }
  } catch (error) {
    console.error('Cache API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, key, data, ttl, tags, question, response } = body;

    switch (action) {
      case 'set':
        if (!key || data === undefined) {
          return NextResponse.json({ error: 'Key and data required' }, { status: 400 });
        }
        const success = await serverCache.set(key, data, ttl || 3600, tags || []);
        return NextResponse.json({ success });

      case 'cache-rag':
        if (!question || !response) {
          return NextResponse.json({ error: 'Question and response required' }, { status: 400 });
        }
        const ragSuccess = await ServerAICacheManager.cacheRAGResponse(
          question,
          response,
          ttl || 1800
        );
        return NextResponse.json({ success: ragSuccess });

      case 'invalidate-tags':
        if (!tags || !Array.isArray(tags)) {
          return NextResponse.json({ error: 'Tags array required' }, { status: 400 });
        }
        const removed = await serverCache.invalidateByTags(tags);
        return NextResponse.json({ removed });

      case 'invalidate-doc':
        const filename = body.filename;
        if (!filename) {
          return NextResponse.json({ error: 'Filename required' }, { status: 400 });
        }
        const docRemoved = await ServerAICacheManager.invalidateDocumentCache(filename);
        return NextResponse.json({ removed: docRemoved });

      case 'flush':
        await serverCache.flush();
        return NextResponse.json({ success: true });

      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
    }
  } catch (error) {
    console.error('Cache API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const key = searchParams.get('key');

    if (!key) {
      return NextResponse.json({ error: 'Key required' }, { status: 400 });
    }

    // Para DELETE, vamos invalidar por tag ou chave específica
    const removed = await serverCache.invalidateByTags([key]);
    return NextResponse.json({ removed });
  } catch (error) {
    console.error('Cache API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
