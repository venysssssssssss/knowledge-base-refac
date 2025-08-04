import React, { useState } from 'react';

const ChatContainer = () => {
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (message: string) => {
    if (!message.trim()) return;

    const newUserMessage = {
      id: Date.now().toString(),
      content: message,
      role: 'user' as const,
      createdAt: new Date(),
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setCurrentMessage('');
    setIsLoading(true);

    try {
      // Send request to RAG service via our proxy
      const response = await fetch('/api/ai/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message, // RAG service expects 'question' but we map it in the proxy
          max_tokens: 512,
          temperature: 0.7,
          search_limit: 3,
          score_threshold: 0.7,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      // RAG service returns: { question, answer, sources, tokens_used, processing_time, ... }
      const aiResponse = {
        id: `ai-${Date.now()}`,
        content: data.answer || 'Desculpe, não consegui processar sua pergunta.',
        role: 'assistant' as const,
        createdAt: new Date(),
        metadata: {
          sources: data.sources || [],
          tokens_used: data.tokens_used || 0,
          processing_time: data.processing_time || 0,
          search_time: data.search_time || 0,
          generation_time: data.generation_time || 0,
        },
      };

      setMessages((prev) => [...prev, aiResponse]);
    } catch (error) {
      console.error('Failed to send message:', error);

      const errorMessage = {
        id: `error-${Date.now()}`,
        content: `Erro: ${error instanceof Error ? error.message : 'Erro desconhecido'}. Verifique se os serviços de IA estão funcionando.`,
        role: 'system' as const,
        createdAt: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      {/* Render your chat UI here */}
    </div>
  );
};

export default ChatContainer;