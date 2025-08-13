import { NextRequest, NextResponse } from 'next/server';

// AI Services configuration
const AI_SERVICES = {
  rag: process.env.RAG_SERVICE_URL || 'http://localhost:8002',
  document: process.env.DOCUMENT_SERVICE_URL || 'http://localhost:8001',
  mistral: process.env.MISTRAL_SERVICE_URL || 'http://localhost:8003',
};

// Simple auth check function - implement your actual auth logic here
async function checkAuth(request: NextRequest): Promise<boolean> {
  // For now, return true to allow all requests
  return true;
}

// Function to proxy requests to AI services
async function proxyToAI(serviceUrl: string, request: NextRequest, endpoint: string): Promise<NextResponse> {
  const url = new URL(`${serviceUrl}${endpoint}`);
  
  // Forward query parameters
  request.nextUrl.searchParams.forEach((value, key) => {
    url.searchParams.append(key, value);
  });
  
  // Forward request with same method and body
  try {
    const response = await fetch(url.toString(), {
      method: request.method,
      headers: {
        'Content-Type': 'application/json',
      },
      body: request.method !== 'GET' ? await request.text() : undefined,
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ AI service responded with ${response.status}: ${errorText}`);
      return NextResponse.json(
        { error: `AI service error: ${response.statusText}`, details: errorText },
        { status: response.status }
      );
    }
    
    const data = await response.text();
    return new NextResponse(data, {
      status: response.status,
      headers: {
        'Content-Type': response.headers.get('Content-Type') || 'application/json',
      },
    });
  } catch (error) {
    console.error(`❌ Proxy Error:`, error);
    return NextResponse.json(
      { 
        error: 'Failed to connect to AI service',
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 502 }
    );
  }
}

async function handleAIRequest(
  request: NextRequest,
  pathSegments: string[]
): Promise<NextResponse> {
  // Verificar autenticação
  const isAuthenticated = await checkAuth(request);
  if (!isAuthenticated) {
    return NextResponse.json(
      { error: 'Unauthorized' },
      { status: 401 }
    );
  }

  const [service, ...endpointParts] = pathSegments;
  const endpoint = endpointParts.length > 0 ? `/${endpointParts.join('/')}` : '';

  console.log(`🔄 Proxying ${request.method} to service: ${service}, endpoint: ${endpoint}`);

  // Roteamento para diferentes serviços
  switch (service) {
    case 'query':
      // RAG queries - redireciona para o RAG service usando endpoint /ask
      return proxyToAI(AI_SERVICES.rag, request, '/ask');
      
    case 'query-full':
      // RAG queries with full context - usa todo o manual como contexto
      return proxyToAI(AI_SERVICES.rag, request, '/ask-full-context');
      
    case 'upload':
      // Document upload - redireciona para document processor
      return proxyToAI(AI_SERVICES.document, request, '/upload-pdf');
      
    case 'search':
      // Document search - redireciona para document processor  
      return proxyToAI(AI_SERVICES.document, request, '/search');
      
    case 'health':
      // Health check - verifica todos os serviços
      return checkAllServicesHealth();
      
    case 'mistral':
      // Direct Mistral access - usa /query para mistral
      const mistralEndpoint = endpoint || '/query';
      return proxyToAI(AI_SERVICES.mistral, request, mistralEndpoint);
      
    case 'mistral-full':
      // Direct Mistral access with full context
      return proxyToAI(AI_SERVICES.mistral, request, '/query-full-context');
      
    case 'rag':
      // Direct RAG access - mapeia /query para /ask
      const ragEndpoint = endpoint === '/query' ? '/ask' : (endpoint || '/ask');
      return proxyToAI(AI_SERVICES.rag, request, ragEndpoint);
      
    case 'document':
      // Direct document processor access
      return proxyToAI(AI_SERVICES.document, request, endpoint);
      
    default:
      return NextResponse.json(
        { 
          error: 'Unknown AI service', 
          available: ['query', 'query-full', 'upload', 'search', 'health', 'mistral', 'mistral-full', 'rag', 'document']
        },
        { status: 404 }
      );
  }
}

// Health check para todos os serviços
async function checkAllServicesHealth(): Promise<NextResponse> {
  const healthChecks = await Promise.allSettled([
    fetch(`${AI_SERVICES.rag}/health`).then(r => ({ service: 'rag', status: r.status, ok: r.ok })),
    fetch(`${AI_SERVICES.document}/health`).then(r => ({ service: 'document', status: r.status, ok: r.ok })),
    fetch(`${AI_SERVICES.mistral}/health`).then(r => ({ service: 'mistral', status: r.status, ok: r.ok }))
  ]);

  const results = healthChecks.map((check, index) => {
    const services = ['rag', 'document', 'mistral'];
    if (check.status === 'fulfilled') {
      return check.value;
    } else {
      return {
        service: services[index],
        status: 0,
        ok: false,
        error: check.reason instanceof Error ? check.reason.message : 'Unknown error'
      };
    }
  });

  const allHealthy = results.every(r => r.ok);

  return NextResponse.json({
    healthy: allHealthy,
    services: results,
    timestamp: new Date().toISOString()
  }, {
    status: allHealthy ? 200 : 503
  });
}

// Export HTTP methods
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const resolvedParams = await params;
  return handleAIRequest(request, resolvedParams.path);
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const resolvedParams = await params;
  return handleAIRequest(request, resolvedParams.path);
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const resolvedParams = await params;
  return handleAIRequest(request, resolvedParams.path);
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const resolvedParams = await params;
  return handleAIRequest(request, resolvedParams.path);
}
