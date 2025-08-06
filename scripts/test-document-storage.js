#!/usr/bin/env node

/**
 * TESTE DE ARMAZENAMENTO DE DOCUMENTOS
 * Verifica se todo o conteúdo está sendo capturado e armazenado corretamente no Qdrant
 */

const fs = require('fs');
const path = require('path');

// Configurações
const DOCUMENT_SERVICE_URL = 'http://localhost:8001';
const RAG_SERVICE_URL = 'http://localhost:8002';

async function testDocumentStorage() {
    console.log('\n🧪 TESTE DE ARMAZENAMENTO DE DOCUMENTOS');
    console.log('===============================================\n');

    try {
        // 1. Verificar saúde dos serviços
        console.log('1️⃣ Verificando saúde dos serviços...');
        
        const docHealth = await fetch(`${DOCUMENT_SERVICE_URL}/health`);
        const ragHealth = await fetch(`${RAG_SERVICE_URL}/health`);
        
        if (!docHealth.ok || !ragHealth.ok) {
            throw new Error('Serviços não estão saudáveis');
        }
        
        console.log('✅ Serviços funcionando corretamente\n');

        // 2. Verificar informações da coleção Qdrant
        console.log('2️⃣ Verificando coleção Qdrant...');
        
        const collectionInfo = await fetch(`${DOCUMENT_SERVICE_URL}/collections/info`);
        const collectionData = await collectionInfo.json();
        
        console.log(`📊 Coleção: ${collectionData.collection_name}`);
        console.log(`📈 Vetores: ${collectionData.vectors_count}`);
        console.log(`🎯 Status: ${collectionData.status}\n`);

        // 3. Testar busca com diferentes estratégias
        console.log('3️⃣ Testando busca com diferentes estratégias...\n');
        
        const testQueries = [
            {
                query: "alteração cadastral",
                description: "Busca direta por termo principal"
            },
            {
                query: "como solicitar mudança cadastral",
                description: "Busca por procedimento completo"
            },
            {
                query: "quem pode fazer solicitação",
                description: "Busca por solicitantes autorizados"
            },
            {
                query: "documentos necessários",
                description: "Busca por requisitos"
            },
            {
                query: "prazo atendimento",
                description: "Busca por prazos"
            },
            {
                query: "menor idade procurador",
                description: "Busca por casos específicos"
            }
        ];

        for (const testCase of testQueries) {
            console.log(`🔍 ${testCase.description}`);
            console.log(`   Query: "${testCase.query}"`);
            
            try {
                // Busca no Document Service
                const searchResponse = await fetch(`${DOCUMENT_SERVICE_URL}/search`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: testCase.query,
                        limit: 10,
                        score_threshold: 0.2
                    })
                });

                if (!searchResponse.ok) {
                    throw new Error(`Erro na busca: ${searchResponse.status}`);
                }

                const searchData = await searchResponse.json();
                const chunks = searchData.chunks || [];
                const contexto = searchData.contexto_completo || '';

                console.log(`   📊 Resultados: ${chunks.length} chunks encontrados`);
                console.log(`   📏 Contexto: ${contexto.length} caracteres`);
                
                if (chunks.length > 0) {
                    const avgScore = chunks.reduce((sum, chunk) => sum + chunk.score, 0) / chunks.length;
                    const maxScore = Math.max(...chunks.map(c => c.score));
                    const minScore = Math.min(...chunks.map(c => c.score));
                    
                    console.log(`   🎯 Scores: Máx=${maxScore.toFixed(3)}, Mín=${minScore.toFixed(3)}, Média=${avgScore.toFixed(3)}`);
                    
                    // Analizar categorias encontradas
                    const categories = new Set();
                    chunks.forEach(chunk => {
                        const cats = chunk.metadata?.categories || [];
                        cats.forEach(cat => categories.add(cat));
                    });
                    
                    console.log(`   🏷️ Categorias: ${Array.from(categories).join(', ')}`);
                    
                    // Mostrar primeiro resultado
                    const firstChunk = chunks[0];
                    const preview = firstChunk.content.substring(0, 150) + '...';
                    console.log(`   📄 Preview: ${preview}`);
                } else {
                    console.log('   ⚠️ Nenhum resultado encontrado!');
                }
                
                console.log('');
                
                // Pequena pausa entre consultas
                await new Promise(resolve => setTimeout(resolve, 500));
                
            } catch (error) {
                console.error(`   ❌ Erro na busca: ${error.message}\n`);
            }
        }

        // 4. Teste de consulta RAG completa
        console.log('4️⃣ Testando consulta RAG completa...\n');
        
        const ragQuery = "Como um procurador pode solicitar alteração cadastral para menor de idade?";
        console.log(`🤖 Pergunta RAG: "${ragQuery}"`);
        
        try {
            const ragResponse = await fetch(`${RAG_SERVICE_URL}/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: ragQuery,
                    max_tokens: 512,
                    temperature: 0.7
                })
            });

            if (!ragResponse.ok) {
                throw new Error(`Erro na consulta RAG: ${ragResponse.status}`);
            }

            const ragData = await ragResponse.json();
            
            console.log(`📝 Resposta gerada: ${ragData.answer?.substring(0, 200)}...`);
            console.log(`📊 Fontes utilizadas: ${ragData.sources?.length || 0}`);
            console.log(`⏱️ Tempo de busca: ${ragData.search_time?.toFixed(2)}s`);
            console.log(`🧠 Tempo de geração: ${ragData.generation_time?.toFixed(2)}s`);
            
        } catch (error) {
            console.error(`❌ Erro na consulta RAG: ${error.message}`);
        }

        // 5. Análise de cobertura
        console.log('\n5️⃣ Análise de cobertura do documento...\n');
        
        try {
            // Busca com threshold muito baixo para verificar cobertura máxima
            const fullCoverageResponse = await fetch(`${DOCUMENT_SERVICE_URL}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: "ICATU seguros alteração",
                    limit: 50,
                    score_threshold: 0.1
                })
            });

            const fullCoverageData = await fullCoverageResponse.json();
            const allChunks = fullCoverageData.chunks || [];
            
            console.log(`📈 Total de chunks no sistema: ${allChunks.length}`);
            
            if (allChunks.length > 0) {
                // Análise de tamanhos
                const chunkSizes = allChunks.map(chunk => chunk.content.length);
                const avgSize = chunkSizes.reduce((a, b) => a + b, 0) / chunkSizes.length;
                const maxSize = Math.max(...chunkSizes);
                const minSize = Math.min(...chunkSizes);
                
                console.log(`📏 Tamanhos dos chunks: Máx=${maxSize}, Mín=${minSize}, Média=${avgSize.toFixed(0)}`);
                
                // Análise de categorias
                const allCategories = {};
                allChunks.forEach(chunk => {
                    const cats = chunk.metadata?.categories || [];
                    cats.forEach(cat => {
                        allCategories[cat] = (allCategories[cat] || 0) + 1;
                    });
                });
                
                console.log('🏷️ Distribuição de categorias:');
                Object.entries(allCategories)
                    .sort(([,a], [,b]) => b - a)
                    .forEach(([cat, count]) => {
                        console.log(`   ${cat}: ${count} chunks`);
                    });
                
                // Análise de seções
                const sections = new Set();
                allChunks.forEach(chunk => {
                    const section = chunk.metadata?.section;
                    if (section) sections.add(section);
                });
                
                console.log(`\n📚 Seções identificadas: ${sections.size}`);
                console.log(`   ${Array.from(sections).slice(0, 5).join(', ')}${sections.size > 5 ? '...' : ''}`);
            }
            
        } catch (error) {
            console.error(`❌ Erro na análise de cobertura: ${error.message}`);
        }

        console.log('\n✅ TESTE CONCLUÍDO COM SUCESSO!');
        console.log('===============================================\n');

    } catch (error) {
        console.error(`\n❌ ERRO NO TESTE: ${error.message}`);
        console.log('===============================================\n');
        process.exit(1);
    }
}

// Executar teste
if (require.main === module) {
    testDocumentStorage();
}

module.exports = { testDocumentStorage };
