import { openDB, DBSchema, IDBPDatabase } from 'idb';

interface Message {
    id: string;
    text: string;
    sender: 'user' | 'bot';
    timestamp: Date;
    attachments?: Attachment[];
    isProcessing?: boolean;
}

interface Attachment {
    id: string;
    name: string;
    type: 'pdf' | 'image' | 'document' | 'spreadsheet' | 'presentation';
    size: number;
    progress?: number;
    extension?: string;
}

interface AttachmentWithData extends Attachment {
    fileData: string;
}

interface ChatDBSchema extends DBSchema {
    messages: {
        key: string;
        value: Message;
        indexes: { 'by-timestamp': Date };
    };
    attachments: {
        key: string;
        value: AttachmentWithData;
    };
}

const DB_NAME = 'ChatCache';
const DB_VERSION = 1;

let dbPromise: Promise<IDBPDatabase<ChatDBSchema>>;

const getDB = () => {
    if (!dbPromise) {
        dbPromise = openDB<ChatDBSchema>(DB_NAME, DB_VERSION, {
            upgrade(db) {
                if (!db.objectStoreNames.contains('messages')) {
                    const messagesStore = db.createObjectStore('messages', { keyPath: 'id' });
                    messagesStore.createIndex('by-timestamp', 'timestamp');
                }
                if (!db.objectStoreNames.contains('attachments')) {
                    db.createObjectStore('attachments', { keyPath: 'id' });
                }
            },
        });
    }
    return dbPromise;
};

export const saveMessageToCache = async (message: Message) => {
    const db = await getDB();
    const tx = db.transaction(['messages', 'attachments'], 'readwrite');

    await tx.objectStore('messages').put(message);

    if (message.attachments) {
        for (const attachment of message.attachments) {
            const attachmentWithData: AttachmentWithData = {
                ...attachment,
                fileData: (attachment as any).fileData || '',
            };
            await tx.objectStore('attachments').put(attachmentWithData);
        }
    }

    await tx.done;
};

export const loadCachedMessages = async (): Promise<Message[]> => {
    const db = await getDB();
    return db.getAll('messages');
};

export const removeFromCache = async (messageId: string) => {
    const db = await getDB();
    const tx = db.transaction(['messages', 'attachments'], 'readwrite');

    const message = await tx.objectStore('messages').get(messageId);
    if (message?.attachments) {
        for (const attachment of message.attachments) {
            await tx.objectStore('attachments').delete(attachment.id);
        }
    }

    await tx.objectStore('messages').delete(messageId);
    await tx.done;
};

export const checkConnectionStatus = () => navigator.onLine;

export const setupConnectionListener = (callback: (isOnline: boolean) => void) => {
    const handleOnline = () => callback(true);
    const handleOffline = () => callback(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
        window.removeEventListener('online', handleOnline);
        window.removeEventListener('offline', handleOffline);
    };
};

export const processFileForOffline = async (file: File): Promise<AttachmentWithData> => {
    const fileData = await readFileAsDataURL(file);
    return {
        id: Date.now().toString(),
        name: file.name,
        type: getFileType(file),
        size: file.size,
        extension: getFileExtension(file.name),
        fileData,
    };
};

const readFileAsDataURL = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
};

const getFileType = (file: File): Attachment['type'] => {
    if (file.type.includes('pdf')) return 'pdf';
    if (file.type.includes('image')) return 'image';
    if (file.type.includes('spreadsheet') || file.type.includes('excel')) return 'spreadsheet';
    if (file.type.includes('presentation') || file.type.includes('powerpoint')) return 'presentation';
    if (file.type.includes('word') || file.type.includes('document')) return 'document';
    return 'document';
};

const getFileExtension = (fileName: string): string => {
    return fileName.split('.').pop()?.toLowerCase() || '';
};