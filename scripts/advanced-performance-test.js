const fetch = require('node-fetch');
const fs = require('fs');

class AdvancedChatPerformanceTester {
  constructor(baseUrl = 'http://localhost:3000/api') {
    this.baseUrl = baseUrl;
    this.testQuestions = [
      {
        question: "Quem pode solicitar uma alteração cadastral na ICATU?",
        category: "solicitante"
      },
      {
        question: "O que devo fazer se os dados no Zendesk estiverem diferentes dos dados no sistema?",
        category: "zendesk"
      },
      {
        question: "Em quanto tempo a alteração cadastral é refletida no Zendesk?",
        category: "prazo_zendesk"
      },
      {
        question: "Qual o prazo máximo para concluir uma solicitação de alteração cadastral?",
        category: "prazo_geral"
      },
      {
        question: "Como registrar uma alteração cadastral no sistema?",
        category: "registro_sistema"
      },
      {
        question: "Quais são os canais disponíveis para envio de documentos?",
        category: "canais_envio"
      },
      {
        question: "Quais documentos são necessários para alterar o nome no cadastro?",
        category: "documentos_nome"
      },
      {
        question: "Como proceder se o cliente afirma que o nome está incorreto, mas foi preenchido corretamente na proposta?",
        category: "nome_incorreto"
      },
      {
        question: "Um cliente quer alterar o estado civil. Quais documentos ele deve apresentar?",
        category: "estado_civil"
      },
      {
        question: "É necessário apresentar algum documento para alterar o nome social?",
        category: "nome_social"
      },
      {
        question: "Um procurador pode solicitar a inclusão do nome social em nome do titular?",
        category: "procurador_nome_social"
      },
      {
        question: "Um cliente menor de idade quer atualizar o nome. Quem pode fazer a solicitação?",
        category: "menor_idade"
      }
    ];
    
    this.concurrencyLevels = [10, 20, 30, 40, 50, 60, 70];
    this.allResults = [];
  }

  generateUniqueId() {
    return `test-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  async makeRequest(question, userId) {
    const startTime = performance.now();
    
    try {
      const response = await fetch(`${this.baseUrl}/ai/rag/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question.question,
          max_tokens: 200,
          temperature: 0.7,
          user_id: userId
        }),
        timeout: 60000 // 60 segundos timeout
      });

      const endTime = performance.now();
      const responseTime = Math.round(endTime - startTime);

      if (!response.ok) {
        return {
          success: false,
          error: `HTTP ${response.status}`,
          responseTime,
          userId,
          question: question.question,
          category: question.category
        };
      }

      const data = await response.json();
      const aiResponse = data.answer || data.response || data.message || '';

      // Cálculo de qualidade da resposta
      const quality = this.calculateResponseQuality(aiResponse);

      return {
        success: true,
        responseTime,
        userId,
        question: question.question,
        category: question.category,
        response: aiResponse,
        responseLength: aiResponse.length,
        tokensUsed: data.tokens_used || data.tokensUsed || 0,
        quality: quality
      };

    } catch (error) {
      const endTime = performance.now();
      const responseTime = Math.round(endTime - startTime);
      
      return {
        success: false,
        error: error.message,
        responseTime,
        userId,
        question: question.question,
        category: question.category
      };
    }
  }

  calculateResponseQuality(response) {
    if (!response) return 0;
    
    const responseText = response.toLowerCase();
    
    // Penaliza respostas genéricas
    const genericPhrases = [
      'não posso oferecer ajuda',
      'contexto fornecido não contém',
      'informações suficientes',
      'não está disponível',
      'desculpe, mas não consigo'
    ];
    
    const isGeneric = genericPhrases.some(phrase => responseText.includes(phrase));
    if (isGeneric) return 25; // Resposta genérica = qualidade baixa
    
    // Bonus por resposta substantiva
    let quality = 50;
    if (response.length > 100) quality += 20;
    if (response.length > 200) quality += 15;
    if (response.length > 300) quality += 10;
    
    // Bonus por conteúdo relevante
    const relevantWords = ['icatu', 'cadastral', 'documento', 'alteração', 'solicitação'];
    const foundRelevant = relevantWords.filter(word => responseText.includes(word)).length;
    quality += foundRelevant * 3;
    
    return Math.min(100, quality);
  }

  async runConcurrentTest(concurrentUsers) {
    console.log(`\n🚀 Iniciando teste com ${concurrentUsers} usuários simultâneos...`);
    
    const startTime = performance.now();
    
    // Criar requests para todos os usuários
    const allRequests = [];
    
    for (let i = 0; i < concurrentUsers; i++) {
      const userId = this.generateUniqueId();
      const randomQuestion = this.testQuestions[Math.floor(Math.random() * this.testQuestions.length)];
      
      allRequests.push(this.makeRequest(randomQuestion, userId));
    }
    
    // Executar todas as requests simultaneamente
    const results = await Promise.allSettled(allRequests);
    
    const endTime = performance.now();
    const totalDuration = (endTime - startTime) / 1000; // em segundos
    
    // Processar resultados
    const successfulResults = results
      .filter(result => result.status === 'fulfilled' && result.value.success)
      .map(result => result.value);
    
    const failedResults = results
      .filter(result => result.status === 'rejected' || (result.status === 'fulfilled' && !result.value.success))
      .map(result => result.status === 'fulfilled' ? result.value : { error: result.reason.message });
    
    const responseTimes = successfulResults.map(r => r.responseTime);
    const qualities = successfulResults.map(r => r.quality);
    const tokenCounts = successfulResults.map(r => r.tokensUsed);
    
    // Calcular estatísticas
    const avgResponseTime = responseTimes.length > 0 ? Math.round(responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length) : 0;
    const medianResponseTime = responseTimes.length > 0 ? this.calculatePercentile(responseTimes, 50) : 0;
    const p95ResponseTime = responseTimes.length > 0 ? this.calculatePercentile(responseTimes, 95) : 0;
    const p99ResponseTime = responseTimes.length > 0 ? this.calculatePercentile(responseTimes, 99) : 0;
    const avgQuality = qualities.length > 0 ? Math.round(qualities.reduce((a, b) => a + b, 0) / qualities.length) : 0;
    const avgTokens = tokenCounts.length > 0 ? Math.round(tokenCounts.reduce((a, b) => a + b, 0) / tokenCounts.length) : 0;
    
    const successRate = (successfulResults.length / concurrentUsers) * 100;
    const errorRate = 100 - successRate;
    const throughput = concurrentUsers / totalDuration;
    
    // Análise por categoria
    const categoryStats = {};
    successfulResults.forEach(result => {
      if (!categoryStats[result.category]) {
        categoryStats[result.category] = {
          count: 0,
          totalResponseTime: 0,
          totalQuality: 0
        };
      }
      categoryStats[result.category].count++;
      categoryStats[result.category].totalResponseTime += result.responseTime;
      categoryStats[result.category].totalQuality += result.quality;
    });
    
    Object.keys(categoryStats).forEach(category => {
      const stats = categoryStats[category];
      stats.avgResponseTime = Math.round(stats.totalResponseTime / stats.count);
      stats.avgQuality = Math.round(stats.totalQuality / stats.count);
    });
    
    const testResult = {
      concurrentUsers,
      successfulRequests: successfulResults.length,
      failedRequests: failedResults.length,
      successRate: Math.round(successRate * 100) / 100,
      errorRate: Math.round(errorRate * 100) / 100,
      totalDuration: Math.round(totalDuration * 100) / 100,
      throughput: Math.round(throughput * 100) / 100,
      avgResponseTime,
      medianResponseTime,
      p95ResponseTime,
      p99ResponseTime,
      avgQuality,
      avgTokens,
      categoryStats,
      errors: failedResults.map(r => r.error),
      timestamp: new Date().toISOString()
    };
    
    console.log(`✅ Concluído: ${successfulResults.length}/${concurrentUsers} sucessos | ⏱️ ${avgResponseTime}ms avg | 🎯 ${avgQuality}% qualidade | 🚀 ${throughput.toFixed(2)} req/s`);
    
    return testResult;
  }

  calculatePercentile(arr, percentile) {
    const sorted = [...arr].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[index] || 0;
  }

  async runFullTestSuite() {
    console.log('🎯 Advanced Chat Performance Tester v2.0');
    console.log('═══════════════════════════════════════════════════════════════════════');
    console.log(`📋 Níveis de concorrência: ${this.concurrencyLevels.join(', ')}`);
    console.log(`❓ Perguntas disponíveis: ${this.testQuestions.length}`);
    console.log(`🕐 Iniciado em: ${new Date().toLocaleString('pt-BR')}`);
    console.log('\n🧪 Executando bateria completa de testes...');

    const allResults = [];
    
    for (let i = 0; i < this.concurrencyLevels.length; i++) {
      const concurrency = this.concurrencyLevels[i];
      
      try {
        const result = await this.runConcurrentTest(concurrency);
        allResults.push(result);
        
        // Pausa entre testes para não sobrecarregar o sistema
        if (i < this.concurrencyLevels.length - 1) {
          console.log(`⏳ Aguardando 3 segundos antes do próximo teste...`);
          await new Promise(resolve => setTimeout(resolve, 3000));
        }
      } catch (error) {
        console.error(`❌ Erro no teste com ${concurrency} usuários:`, error.message);
        allResults.push({
          concurrentUsers: concurrency,
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    }

    this.allResults = allResults;
    
    // Gerar relatórios
    const summary = this.generateSummary();
    const markdownReport = this.generateMarkdownReport();
    
    // Salvar arquivos
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const jsonFilename = `chat-performance-advanced-${timestamp}.json`;
    const mdFilename = `chat-performance-report-${timestamp}.md`;
    
    fs.writeFileSync(jsonFilename, JSON.stringify({
      summary,
      detailedResults: allResults,
      metadata: {
        testDate: new Date().toISOString(),
        totalTests: this.concurrencyLevels.length,
        questionsUsed: this.testQuestions.length
      }
    }, null, 2));
    
    fs.writeFileSync(mdFilename, markdownReport);
    
    console.log('\n🏁 TESTE COMPLETO FINALIZADO!');
    console.log('═══════════════════════════════════════════════════════════════════════');
    console.log(`📊 Relatório Markdown: ${mdFilename}`);
    console.log(`💾 Dados JSON: ${jsonFilename}`);
    console.log('✨ Todos os testes concluídos com sucesso!');
    
    return {
      summary,
      allResults,
      markdownReport,
      files: { json: jsonFilename, markdown: mdFilename }
    };
  }

  generateSummary() {
    const validResults = this.allResults.filter(r => !r.error);
    
    if (validResults.length === 0) {
      return { error: 'Nenhum teste válido executado' };
    }
    
    const bestThroughput = Math.max(...validResults.map(r => r.throughput));
    const bestQuality = Math.max(...validResults.map(r => r.avgQuality));
    const avgSuccessRate = validResults.reduce((sum, r) => sum + r.successRate, 0) / validResults.length;
    const totalRequests = validResults.reduce((sum, r) => sum + r.concurrentUsers, 0);
    const totalSuccessful = validResults.reduce((sum, r) => sum + r.successfulRequests, 0);
    
    return {
      testsExecuted: validResults.length,
      totalRequests,
      totalSuccessful,
      overallSuccessRate: Math.round(avgSuccessRate * 100) / 100,
      bestThroughput: Math.round(bestThroughput * 100) / 100,
      bestQuality,
      recommendedConcurrency: validResults.find(r => r.throughput === bestThroughput)?.concurrentUsers || 'N/A'
    };
  }

  generateMarkdownReport() {
    const timestamp = new Date().toLocaleString('pt-BR');
    const summary = this.generateSummary();
    const validResults = this.allResults.filter(r => !r.error);
    
    let markdown = `# 📊 Chat Performance Test Report\n\n`;
    markdown += `**Data do Teste:** ${timestamp}  \n`;
    markdown += `**Sistema:** ICATU Knowledge Base  \n`;
    markdown += `**Testes Executados:** ${summary.testsExecuted}  \n`;
    markdown += `**Total de Requests:** ${summary.totalRequests}  \n\n`;
    
    markdown += `## 🎯 Resumo Executivo\n\n`;
    markdown += `| Métrica | Valor |\n`;
    markdown += `|---------|-------|\n`;
    markdown += `| 📈 **Taxa de Sucesso Geral** | **${summary.overallSuccessRate}%** |\n`;
    markdown += `| 🚀 **Melhor Throughput** | **${summary.bestThroughput} req/s** |\n`;
    markdown += `| 🎯 **Melhor Qualidade** | **${summary.bestQuality}%** |\n`;
    markdown += `| ⚡ **Concorrência Recomendada** | **${summary.recommendedConcurrency} usuários** |\n`;
    markdown += `| ✅ **Requests Bem-sucedidos** | **${summary.totalSuccessful}/${summary.totalRequests}** |\n\n`;
    
    markdown += `## 📋 Resultados Detalhados\n\n`;
    markdown += `| Usuários | ✅ Taxa Sucesso | ⏱️ Tempo Médio | 📊 P95 | 📈 P99 | 🎯 Qualidade | 🚀 Throughput | ⏰ Duração |\n`;
    markdown += `|----------|----------------|----------------|-------|-------|-------------|-------------|------------|\n`;
    
    validResults.forEach(result => {
      markdown += `| **${result.concurrentUsers}** | ${result.successRate}% | ${result.avgResponseTime}ms | ${result.p95ResponseTime}ms | ${result.p99ResponseTime}ms | ${result.avgQuality}% | ${result.throughput.toFixed(2)} req/s | ${result.totalDuration}s |\n`;
    });
    
    markdown += `\n## 📈 Análise de Performance\n\n`;
    
    // Gráfico ASCII simples
    markdown += `### 🚀 Throughput por Concorrência\n\n`;
    markdown += `\`\`\`\n`;
    validResults.forEach(result => {
      const barLength = Math.round((result.throughput / summary.bestThroughput) * 40);
      const bar = '█'.repeat(barLength) + '░'.repeat(40 - barLength);
      markdown += `${result.concurrentUsers.toString().padStart(3)}u │${bar}│ ${result.throughput.toFixed(2)} req/s\n`;
    });
    markdown += `\`\`\`\n\n`;
    
    // Análise de qualidade
    markdown += `### 🎯 Qualidade das Respostas\n\n`;
    markdown += `\`\`\`\n`;
    validResults.forEach(result => {
      const barLength = Math.round((result.avgQuality / 100) * 40);
      const bar = '█'.repeat(barLength) + '░'.repeat(40 - barLength);
      markdown += `${result.concurrentUsers.toString().padStart(3)}u │${bar}│ ${result.avgQuality}%\n`;
    });
    markdown += `\`\`\`\n\n`;
    
    // Análise por categoria
    markdown += `### 📝 Performance por Categoria\n\n`;
    const allCategories = new Set();
    validResults.forEach(result => {
      Object.keys(result.categoryStats).forEach(cat => allCategories.add(cat));
    });
    
    if (allCategories.size > 0) {
      markdown += `| Categoria | Teste 10u | Teste 20u | Teste 30u | Teste 40u | Teste 50u | Teste 60u | Teste 70u |\n`;
      markdown += `|-----------|-----------|-----------|-----------|-----------|-----------|-----------|----------|\n`;
      
      [...allCategories].forEach(category => {
        markdown += `| **${category}** |`;
        validResults.forEach(result => {
          const catStats = result.categoryStats[category];
          if (catStats) {
            markdown += ` ${catStats.avgResponseTime}ms (${catStats.avgQuality}%) |`;
          } else {
            markdown += ` - |`;
          }
        });
        markdown += `\n`;
      });
      markdown += `\n`;
    }
    
    // Recomendações
    markdown += `## 💡 Recomendações\n\n`;
    
    const bestResult = validResults.find(r => r.throughput === summary.bestThroughput);
    const worstQuality = Math.min(...validResults.map(r => r.avgQuality));
    
    if (summary.overallSuccessRate >= 95) {
      markdown += `✅ **Sistema Estável:** Taxa de sucesso excelente (${summary.overallSuccessRate}%)  \n`;
    } else if (summary.overallSuccessRate >= 90) {
      markdown += `⚠️ **Sistema Bom:** Taxa de sucesso aceitável (${summary.overallSuccessRate}%)  \n`;
    } else {
      markdown += `🔴 **Atenção:** Taxa de sucesso baixa (${summary.overallSuccessRate}%) - investigar erros  \n`;
    }
    
    if (bestResult) {
      markdown += `🚀 **Ponto Ótimo:** ${bestResult.concurrentUsers} usuários simultâneos (${bestResult.throughput.toFixed(2)} req/s)  \n`;
    }
    
    if (worstQuality < 70) {
      markdown += `🎯 **Qualidade:** Melhorar base de conhecimento (qualidade mínima: ${worstQuality}%)  \n`;
    }
    
    markdown += `\n## 🔧 Configurações do Teste\n\n`;
    markdown += `- **Endpoint:** \`/api/ai/rag/query\`  \n`;
    markdown += `- **Timeout:** 60 segundos  \n`;
    markdown += `- **Pausa entre testes:** 3 segundos  \n`;
    markdown += `- **Perguntas:** ${this.testQuestions.length} categorias diferentes  \n`;
    markdown += `- **Método:** POST com JSON payload  \n\n`;
    
    // Erros encontrados
    const errors = this.allResults.filter(r => r.error);
    if (errors.length > 0) {
      markdown += `## ❌ Erros Encontrados\n\n`;
      errors.forEach(error => {
        markdown += `- **${error.concurrentUsers} usuários:** ${error.error}  \n`;
      });
      markdown += `\n`;
    }
    
    markdown += `---\n`;
    markdown += `*Relatório gerado automaticamente em ${timestamp}*\n`;
    
    return markdown;
  }
}

// Executar se chamado diretamente
if (require.main === module) {
  const tester = new AdvancedChatPerformanceTester();
  tester.runFullTestSuite()
    .then(results => {
      console.log('\n📊 Teste finalizado com sucesso!');
      process.exit(0);
    })
    .catch(error => {
      console.error('💥 Erro no teste:', error.message);
      process.exit(1);
    });
}

module.exports = AdvancedChatPerformanceTester;
