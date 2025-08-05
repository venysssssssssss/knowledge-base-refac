/**
 * Testes de Performance do Sistema de Chat
 * Avalia assertividade, tempo de resposta e simultaneidade
 */

interface TestResult {
  testId: string;
  concurrent_users: number;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  avg_response_time: number;
  min_response_time: number;
  max_response_time: number;
  p95_response_time: number;
  assertiveness_score: number;
  error_rate: number;
  throughput: number; // requests per second
  test_duration: number; // in seconds
  timestamp: Date;
  errors: string[];
}

interface ChatRequest {
  id: string;
  question: string;
  expected_keywords?: string[];
  start_time: number;
  end_time?: number;
  response?: string;
  success?: boolean;
  error?: string;
}

class ChatPerformanceTester {
  private baseUrl: string;
  private testQuestions: Array<{
    question: string;
    expected_keywords: string[];
    category: string;
  }>;

  constructor(baseUrl: string = 'http://localhost:3000') {
    this.baseUrl = baseUrl;
    this.testQuestions = [
      {
        question: "Quem pode solicitar alteração cadastral na ICATU?",
        expected_keywords: ["titular", "apólice", "procurador", "curador", "tutor"],
        category: "cadastro"
      },
      {
        question: "Como fazer upload de documentos?",
        expected_keywords: ["upload", "documento", "pdf", "processar"],
        category: "upload"
      },
      {
        question: "Qual o limite de tamanho para arquivos?",
        expected_keywords: ["tamanho", "limite", "10MB", "arquivo"],
        category: "limite"
      },
      {
        question: "Quais tipos de arquivo são aceitos?",
        expected_keywords: ["pdf", "doc", "xlsx", "ppt", "formato"],
        category: "formato"
      },
      {
        question: "Como funciona a busca por documentos?",
        expected_keywords: ["busca", "documento", "pesquisa", "conteúdo"],
        category: "busca"
      },
      {
        question: "O que é RAG?",
        expected_keywords: ["rag", "retrieval", "geração", "busca"],
        category: "conceito"
      },
      {
        question: "Como cancelar uma solicitação?",
        expected_keywords: ["cancelar", "solicitação", "processo"],
        category: "cancelamento"
      },
      {
        question: "Preciso estar logado para usar o chat?",
        expected_keywords: ["login", "autenticação", "acesso"],
        category: "autenticacao"
      }
    ];
  }

  /**
   * Executa um teste de chat individual
   */
  private async executeSingleChatTest(request: ChatRequest): Promise<ChatRequest> {
    try {
      const response = await fetch(`${this.baseUrl}/api/ai/rag/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: request.question,
          max_tokens: 512,
          temperature: 0.7,
          search_limit: 3,
          score_threshold: 0.6
        })
      });

      request.end_time = Date.now();

      if (!response.ok) {
        const errorText = await response.text();
        request.success = false;
        request.error = `HTTP ${response.status}: ${errorText}`;
        return request;
      }

      const data = await response.json();
      request.response = data.answer || '';
      request.success = true;

      return request;
    } catch (error) {
      request.end_time = Date.now();
      request.success = false;
      request.error = error instanceof Error ? error.message : 'Unknown error';
      return request;
    }
  }

  /**
   * Calcula a assertividade baseada nas palavras-chave esperadas
   */
  private calculateAssertiveness(response: string, expectedKeywords: string[]): number {
    if (!response || expectedKeywords.length === 0) return 0;

    const responseWords = response.toLowerCase().split(/\s+/);
    const foundKeywords = expectedKeywords.filter(keyword => 
      responseWords.some(word => word.includes(keyword.toLowerCase()))
    );

    return (foundKeywords.length / expectedKeywords.length) * 100;
  }

  /**
   * Executa teste de simultaneidade
   */
  async runConcurrentTest(concurrentUsers: number): Promise<TestResult> {
    console.log(`🚀 Iniciando teste com ${concurrentUsers} usuários simultâneos...`);
    
    const testId = `test-${Date.now()}-${concurrentUsers}users`;
    const startTime = Date.now();
    
    // Criar requests para todos os usuários
    const allRequests: ChatRequest[] = [];
    
    for (let i = 0; i < concurrentUsers; i++) {
      const questionData = this.testQuestions[i % this.testQuestions.length];
      allRequests.push({
        id: `req-${i}`,
        question: questionData.question,
        expected_keywords: questionData.expected_keywords,
        start_time: Date.now()
      });
    }

    // Executar todos os requests simultaneamente
    const results = await Promise.allSettled(
      allRequests.map(req => this.executeSingleChatTest(req))
    );

    const endTime = Date.now();
    const testDuration = (endTime - startTime) / 1000;

    // Processar resultados
    const processedRequests = results.map((result, index) => {
      if (result.status === 'fulfilled') {
        return result.value;
      } else {
        return {
          ...allRequests[index],
          end_time: Date.now(),
          success: false,
          error: result.reason?.message || 'Promise rejected'
        };
      }
    });

    // Calcular métricas
    const successfulRequests = processedRequests.filter(req => req.success);
    const failedRequests = processedRequests.filter(req => !req.success);
    
    const responseTimes = successfulRequests
      .map(req => (req.end_time! - req.start_time))
      .sort((a, b) => a - b);

    const assertivenessScores = successfulRequests.map(req => {
      if (!req.expected_keywords || !req.response) return 0;
      return this.calculateAssertiveness(req.response, req.expected_keywords);
    });

    const avgAssertiveness = assertivenessScores.length > 0 
      ? assertivenessScores.reduce((sum, score) => sum + score, 0) / assertivenessScores.length
      : 0;

    const p95Index = Math.floor(responseTimes.length * 0.95);

    const testResult: TestResult = {
      testId,
      concurrent_users: concurrentUsers,
      total_requests: allRequests.length,
      successful_requests: successfulRequests.length,
      failed_requests: failedRequests.length,
      avg_response_time: responseTimes.length > 0 
        ? responseTimes.reduce((sum, time) => sum + time, 0) / responseTimes.length 
        : 0,
      min_response_time: responseTimes.length > 0 ? responseTimes[0] : 0,
      max_response_time: responseTimes.length > 0 ? responseTimes[responseTimes.length - 1] : 0,
      p95_response_time: responseTimes.length > 0 ? responseTimes[p95Index] || 0 : 0,
      assertiveness_score: avgAssertiveness,
      error_rate: (failedRequests.length / allRequests.length) * 100,
      throughput: successfulRequests.length / testDuration,
      test_duration: testDuration,
      timestamp: new Date(),
      errors: failedRequests.map(req => req.error || 'Unknown error')
    };

    return testResult;
  }

  /**
   * Executa bateria completa de testes
   */
  async runFullTestSuite(): Promise<TestResult[]> {
    const concurrentUserCounts = [10, 20, 30, 40, 50];
    const results: TestResult[] = [];

    console.log('🧪 Iniciando bateria completa de testes de performance...\n');

    for (const userCount of concurrentUserCounts) {
      try {
        const result = await this.runConcurrentTest(userCount);
        results.push(result);
        
        this.printTestResult(result);
        
        // Esperar um pouco entre os testes para não sobrecarregar
        if (userCount < 50) {
          console.log('⏳ Aguardando 5 segundos antes do próximo teste...\n');
          await new Promise(resolve => setTimeout(resolve, 5000));
        }
      } catch (error) {
        console.error(`❌ Erro no teste com ${userCount} usuários:`, error);
      }
    }

    this.printSummary(results);
    return results;
  }

  /**
   * Imprime resultado de um teste individual
   */
  private printTestResult(result: TestResult) {
    console.log(`📊 Resultado do Teste - ${result.concurrent_users} usuários simultâneos`);
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log(`✅ Requests bem-sucedidos: ${result.successful_requests}/${result.total_requests}`);
    console.log(`❌ Taxa de erro: ${result.error_rate.toFixed(2)}%`);
    console.log(`⏱️  Tempo médio de resposta: ${result.avg_response_time.toFixed(0)}ms`);
    console.log(`📈 P95 tempo de resposta: ${result.p95_response_time.toFixed(0)}ms`);
    console.log(`🎯 Assertividade média: ${result.assertiveness_score.toFixed(1)}%`);
    console.log(`🚀 Throughput: ${result.throughput.toFixed(2)} req/s`);
    console.log(`⏰ Duração do teste: ${result.test_duration.toFixed(2)}s`);
    
    if (result.errors.length > 0) {
      console.log(`\n🔴 Erros encontrados:`);
      result.errors.slice(0, 3).forEach((error, index) => {
        console.log(`   ${index + 1}. ${error}`);
      });
      if (result.errors.length > 3) {
        console.log(`   ... e mais ${result.errors.length - 3} erros`);
      }
    }
    console.log('\n');
  }

  /**
   * Imprime resumo de todos os testes
   */
  private printSummary(results: TestResult[]) {
    console.log('\n🏁 RESUMO GERAL DOS TESTES');
    console.log('═══════════════════════════════════════════════════════════════════════');
    
    console.log('\n📋 Tabela de Resultados:');
    console.log('Users | Success Rate | Avg Response | P95 Response | Assertiveness | Throughput');
    console.log('------|--------------|--------------|--------------|---------------|------------');
    
    results.forEach(result => {
      const successRate = ((result.successful_requests / result.total_requests) * 100).toFixed(1);
      console.log(
        `${result.concurrent_users.toString().padStart(5)} | ` +
        `${successRate.padStart(11)}% | ` +
        `${result.avg_response_time.toFixed(0).padStart(11)}ms | ` +
        `${result.p95_response_time.toFixed(0).padStart(11)}ms | ` +
        `${result.assertiveness_score.toFixed(1).padStart(12)}% | ` +
        `${result.throughput.toFixed(2).padStart(9)} req/s`
      );
    });

    // Análise de performance
    console.log('\n📈 Análise de Performance:');
    
    const bestThroughput = Math.max(...results.map(r => r.throughput));
    const bestThroughputTest = results.find(r => r.throughput === bestThroughput);
    
    const bestAssertiveness = Math.max(...results.map(r => r.assertiveness_score));
    const bestAssertivenessTest = results.find(r => r.assertiveness_score === bestAssertiveness);
    
    console.log(`🚀 Melhor throughput: ${bestThroughput.toFixed(2)} req/s (${bestThroughputTest?.concurrent_users} usuários)`);
    console.log(`🎯 Melhor assertividade: ${bestAssertiveness.toFixed(1)}% (${bestAssertivenessTest?.concurrent_users} usuários)`);
    
    const avgErrorRate = results.reduce((sum, r) => sum + r.error_rate, 0) / results.length;
    console.log(`⚠️  Taxa média de erro: ${avgErrorRate.toFixed(2)}%`);
    
    console.log('\n✨ Teste de performance concluído!');
  }

  /**
   * Salva resultados em arquivo JSON
   */
  async saveResults(results: TestResult[], filename?: string): Promise<void> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const fname = filename || `chat-performance-test-${timestamp}.json`;
    
    const fs = await import('fs/promises');
    await fs.writeFile(fname, JSON.stringify(results, null, 2));
    console.log(`💾 Resultados salvos em: ${fname}`);
  }
}

export { ChatPerformanceTester, type TestResult, type ChatRequest };

// Função para executar os testes via CLI ou import
export async function runPerformanceTests() {
  const tester = new ChatPerformanceTester();
  const results = await tester.runFullTestSuite();
  
  // Salvar resultados
  try {
    await tester.saveResults(results);
  } catch (error) {
    console.warn('⚠️ Não foi possível salvar os resultados:', error);
  }
  
  return results;
}

// Se executado diretamente
if (require.main === module) {
  runPerformanceTests().catch(console.error);
}
