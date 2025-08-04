/**
 * Sistema de Filas para Processamento de Perguntas Similares
 * Implementa queue com debouncing e batching para otimizar processamento de IA
 */

type QueueJobStatus = 'pending' | 'processing' | 'completed' | 'failed';

interface QueueJob<T = any> {
  id: string;
  type: string;
  data: T;
  priority: number;
  status: QueueJobStatus;
  retries: number;
  maxRetries: number;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  result?: any;
  error?: string;
  dependsOn?: string[]; // IDs de jobs que devem ser executados antes
}

interface QueueConfig {
  maxConcurrent: number;
  retryDelay: number;
  maxRetries: number;
  batchSize: number;
  batchTimeout: number;
}

interface QueueStats {
  pending: number;
  processing: number;
  completed: number;
  failed: number;
  totalProcessed: number;
  avgProcessingTime: number;
}

type JobProcessor<T = any, R = any> = (job: QueueJob<T>) => Promise<R>;
type BatchProcessor<T = any, R = any> = (jobs: QueueJob<T>[]) => Promise<R[]>;

class SmartQueue {
  private jobs = new Map<string, QueueJob>();
  private processors = new Map<string, JobProcessor>();
  private batchProcessors = new Map<string, BatchProcessor>();
  private processingJobs = new Set<string>();
  private config: QueueConfig;
  private stats: QueueStats;
  private batchTimers = new Map<string, NodeJS.Timeout>();

  constructor(config: Partial<QueueConfig> = {}) {
    this.config = {
      maxConcurrent: config.maxConcurrent || 5,
      retryDelay: config.retryDelay || 1000,
      maxRetries: config.maxRetries || 3,
      batchSize: config.batchSize || 10,
      batchTimeout: config.batchTimeout || 2000,
      ...config
    };

    this.stats = {
      pending: 0,
      processing: 0,
      completed: 0,
      failed: 0,
      totalProcessed: 0,
      avgProcessingTime: 0
    };

    this.startProcessor();
  }

  /**
   * Registra um processador para um tipo de job
   */
  registerProcessor<T, R>(type: string, processor: JobProcessor<T, R>) {
    this.processors.set(type, processor as JobProcessor);
    console.log(`📝 Queue: Processador registrado para tipo "${type}"`);
  }

  /**
   * Registra um processador em lote para um tipo de job
   */
  registerBatchProcessor<T, R>(type: string, processor: BatchProcessor<T, R>) {
    this.batchProcessors.set(type, processor as BatchProcessor);
    console.log(`📦 Queue: Processador em lote registrado para tipo "${type}"`);
  }

  /**
   * Adiciona um job à fila
   */
  async addJob<T>(
    type: string,
    data: T,
    options: {
      priority?: number;
      maxRetries?: number;
      dependsOn?: string[];
      id?: string;
    } = {}
  ): Promise<string> {
    const id = options.id || this.generateJobId();
    
    const job: QueueJob<T> = {
      id,
      type,
      data,
      priority: options.priority || 0,
      status: 'pending',
      retries: 0,
      maxRetries: options.maxRetries || this.config.maxRetries,
      createdAt: new Date(),
      dependsOn: options.dependsOn || []
    };

    this.jobs.set(id, job);
    this.stats.pending++;

    console.log(`➕ Queue: Job adicionado [${type}] - ${id}`);
    
    // Se há processador em lote, configurar timer
    if (this.batchProcessors.has(type)) {
      this.scheduleBatchProcessing(type);
    }

    // Tentar processar imediatamente
    this.processNext();

    return id;
  }

  /**
   * Adiciona múltiplos jobs similares (para deduplicação)
   */
  async addSimilarJobs<T>(
    type: string,
    jobs: Array<{ data: T; priority?: number; id?: string }>
  ): Promise<string[]> {
    const jobIds: string[] = [];
    
    // Detectar jobs similares e agrupar
    const groupedJobs = this.groupSimilarJobs(jobs);
    
    for (const group of groupedJobs) {
      if (group.length === 1) {
        // Job único
        const jobId = await this.addJob(type, group[0].data, {
          priority: group[0].priority,
          id: group[0].id
        });
        jobIds.push(jobId);
      } else {
        // Jobs similares - criar um job de lote
        const batchId = await this.addJob(`${type}_batch`, group, {
          priority: Math.max(...group.map(j => j.priority || 0))
        });
        jobIds.push(batchId);
      }
    }

    return jobIds;
  }

  /**
   * Agrupa jobs similares para processamento em lote
   */
  private groupSimilarJobs<T>(jobs: Array<{ data: T; priority?: number; id?: string }>): Array<Array<{ data: T; priority?: number; id?: string }>> {
    // Implementação simples - pode ser melhorada com algoritmos de similaridade
    const groups: Array<Array<{ data: T; priority?: number; id?: string }>> = [];
    
    for (const job of jobs) {
      let addedToGroup = false;
      
      for (const group of groups) {
        if (group.length < this.config.batchSize && this.areSimilar(job.data, group[0].data)) {
          group.push(job);
          addedToGroup = true;
          break;
        }
      }
      
      if (!addedToGroup) {
        groups.push([job]);
      }
    }

    return groups;
  }

  /**
   * Verifica se dois jobs são similares (implementação básica)
   */
  private areSimilar(data1: any, data2: any): boolean {
    // Para perguntas de IA, verificar similaridade textual básica
    if (typeof data1 === 'object' && typeof data2 === 'object') {
      if (data1.question && data2.question) {
        const q1 = data1.question.toLowerCase().trim();
        const q2 = data2.question.toLowerCase().trim();
        
        // Similaridade simples baseada em palavras comuns
        const words1 = new Set(q1.split(/\s+/));
        const words2 = new Set(q2.split(/\s+/));
        const intersection = new Set([...words1].filter(w => words2.has(w)));
        const union = new Set([...words1, ...words2]);
        
        const similarity = intersection.size / union.size;
        return similarity > 0.6; // 60% de similaridade
      }
    }
    
    return false;
  }

  /**
   * Programa processamento em lote
   */
  private scheduleBatchProcessing(type: string) {
    // Limpar timer existente
    if (this.batchTimers.has(type)) {
      clearTimeout(this.batchTimers.get(type)!);
    }

    // Criar novo timer
    const timer = setTimeout(() => {
      this.processBatch(type);
      this.batchTimers.delete(type);
    }, this.config.batchTimeout);

    this.batchTimers.set(type, timer);
  }

  /**
   * Processa jobs em lote
   */
  private async processBatch(type: string) {
    const batchProcessor = this.batchProcessors.get(type);
    if (!batchProcessor) return;

    // Buscar jobs pendentes do tipo
    const pendingJobs = Array.from(this.jobs.values())
      .filter(job => job.type === type && job.status === 'pending')
      .sort((a, b) => b.priority - a.priority)
      .slice(0, this.config.batchSize);

    if (pendingJobs.length === 0) return;

    console.log(`📦 Queue: Processando lote de ${pendingJobs.length} jobs [${type}]`);

    // Marcar jobs como processando
    for (const job of pendingJobs) {
      job.status = 'processing';
      job.startedAt = new Date();
      this.processingJobs.add(job.id);
    }

    this.updateStats();

    try {
      const results = await batchProcessor(pendingJobs);
      
      // Marcar como completados
      for (let i = 0; i < pendingJobs.length; i++) {
        const job = pendingJobs[i];
        job.status = 'completed';
        job.completedAt = new Date();
        job.result = results[i];
        this.processingJobs.delete(job.id);
      }

      console.log(`✅ Queue: Lote processado com sucesso [${type}]`);
    } catch (error) {
      console.error(`❌ Queue: Erro no processamento em lote [${type}]:`, error);
      
      // Marcar jobs como falhou
      for (const job of pendingJobs) {
        job.status = 'failed';
        job.error = error instanceof Error ? error.message : 'Unknown error';
        job.completedAt = new Date();
        this.processingJobs.delete(job.id);
      }
    }

    this.updateStats();
  }

  /**
   * Processador principal da fila
   */
  private async startProcessor() {
    setInterval(() => {
      this.processNext();
    }, 100); // Verificar a cada 100ms
  }

  /**
   * Processa próximo job da fila
   */
  private async processNext() {
    if (this.processingJobs.size >= this.config.maxConcurrent) {
      return; // Limite de concorrência atingido
    }

    // Buscar próximo job pronto para processar
    const nextJob = this.getNextReadyJob();
    if (!nextJob) return;

    await this.processJob(nextJob);
  }

  /**
   * Busca próximo job pronto para processamento
   */
  private getNextReadyJob(): QueueJob | null {
    const pendingJobs = Array.from(this.jobs.values())
      .filter(job => {
        if (job.status !== 'pending') return false;
        
        // Verificar dependências
        if (job.dependsOn && job.dependsOn.length > 0) {
          return job.dependsOn.every(depId => {
            const depJob = this.jobs.get(depId);
            return depJob && depJob.status === 'completed';
          });
        }
        
        return true;
      })
      .sort((a, b) => b.priority - a.priority);

    return pendingJobs[0] || null;
  }

  /**
   * Processa um job individual
   */
  private async processJob(job: QueueJob) {
    const processor = this.processors.get(job.type);
    if (!processor) {
      console.warn(`⚠️ Queue: Processador não encontrado para tipo "${job.type}"`);
      return;
    }

    job.status = 'processing';
    job.startedAt = new Date();
    this.processingJobs.add(job.id);
    this.updateStats();

    console.log(`🔄 Queue: Processando job [${job.type}] - ${job.id}`);

    try {
      const result = await processor(job);
      
      job.status = 'completed';
      job.completedAt = new Date();
      job.result = result;
      
      console.log(`✅ Queue: Job completado [${job.type}] - ${job.id}`);
    } catch (error) {
      console.error(`❌ Queue: Erro no job [${job.type}] - ${job.id}:`, error);
      
      job.retries++;
      if (job.retries < job.maxRetries) {
        // Reagendar para retry
        job.status = 'pending';
        setTimeout(() => {
          this.processNext();
        }, this.config.retryDelay * job.retries);
        
        console.log(`🔄 Queue: Reagendando job para retry (${job.retries}/${job.maxRetries})`);
      } else {
        job.status = 'failed';
        job.completedAt = new Date();
        job.error = error instanceof Error ? error.message : 'Unknown error';
      }
    }

    this.processingJobs.delete(job.id);
    this.updateStats();
  }

  /**
   * Atualiza estatísticas
   */
  private updateStats() {
    const allJobs = Array.from(this.jobs.values());
    
    this.stats.pending = allJobs.filter(j => j.status === 'pending').length;
    this.stats.processing = this.processingJobs.size;
    this.stats.completed = allJobs.filter(j => j.status === 'completed').length;
    this.stats.failed = allJobs.filter(j => j.status === 'failed').length;
    this.stats.totalProcessed = this.stats.completed + this.stats.failed;

    // Calcular tempo médio de processamento
    const completedJobs = allJobs.filter(j => j.status === 'completed' && j.startedAt && j.completedAt);
    if (completedJobs.length > 0) {
      const totalTime = completedJobs.reduce((sum, job) => {
        return sum + (job.completedAt!.getTime() - job.startedAt!.getTime());
      }, 0);
      this.stats.avgProcessingTime = totalTime / completedJobs.length;
    }
  }

  /**
   * Obtém status de um job
   */
  getJobStatus(id: string): QueueJob | null {
    return this.jobs.get(id) || null;
  }

  /**
   * Obtém estatísticas da fila
   */
  getStats(): QueueStats {
    this.updateStats();
    return { ...this.stats };
  }

  /**
   * Aguarda conclusão de um job
   */
  async waitForJob(id: string, timeout: number = 30000): Promise<any> {
    return new Promise((resolve, reject) => {
      const checkInterval = setInterval(() => {
        const job = this.jobs.get(id);
        if (!job) {
          clearInterval(checkInterval);
          reject(new Error('Job não encontrado'));
          return;
        }

        if (job.status === 'completed') {
          clearInterval(checkInterval);
          resolve(job.result);
        } else if (job.status === 'failed') {
          clearInterval(checkInterval);
          reject(new Error(job.error || 'Job falhou'));
        }
      }, 100);

      // Timeout
      setTimeout(() => {
        clearInterval(checkInterval);
        reject(new Error('Timeout aguardando job'));
      }, timeout);
    });
  }

  /**
   * Limpa jobs antigos
   */
  cleanup(maxAge: number = 24 * 60 * 60 * 1000) { // 24 horas
    const cutoff = new Date(Date.now() - maxAge);
    const toDelete: string[] = [];

    for (const [id, job] of this.jobs) {
      if (job.completedAt && job.completedAt < cutoff) {
        toDelete.push(id);
      }
    }

    for (const id of toDelete) {
      this.jobs.delete(id);
    }

    if (toDelete.length > 0) {
      console.log(`🧹 Queue: Removidos ${toDelete.length} jobs antigos`);
    }
  }

  private generateJobId(): string {
    return `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

// Instância singleton
export const smartQueue = new SmartQueue({
  maxConcurrent: 3, // Máximo 3 perguntas simultâneas para IA
  batchSize: 5,     // Processar até 5 perguntas similares juntas
  batchTimeout: 3000 // Aguardar 3s por mais perguntas similares
});

export type { QueueJob, QueueConfig, QueueStats, JobProcessor, BatchProcessor };
export { SmartQueue };
