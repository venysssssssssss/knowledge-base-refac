export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  createdAt: Date;
  metadata?: {
    sources?: Array<{
      content_preview: string;
      score: number;
      metadata: Record<string, any>;
    }>;
    tokens_used?: number;
    processing_time?: number;
    search_time?: number;
    generation_time?: number;
  };
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  currentMessage: string;
}
