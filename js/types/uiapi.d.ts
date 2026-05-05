/**
 * TypeScript type definitions for ComfyUI-uiapi internal types
 * These types describe the internal data structures used by the UIAPI extension
 * for communication, connection management, and UI components.
 */

import type { LGraphNode, LGraphWidget, LGraphNodeInput, LGraphNodeOutput } from './comfyui';

// ============================================================================
// API Request/Response Types
// ============================================================================

/**
 * Base interface for all UIAPI requests sent from server to WebUI
 */
export interface UIAPIRequest {
    /** Unique request identifier */
    request_id: string;

    /** Client ID assigned by server */
    client_id: string;

    /** API endpoint being called */
    endpoint: string;

    /** Request data (endpoint-specific) */
    data?: any;

    /** Enable verbose logging for this request */
    verbose?: boolean;
}

/**
 * Base interface for all UIAPI responses sent from WebUI to server
 */
export interface UIAPIResponse {
    /** Response data (endpoint-specific) */
    response?: any;

    /** Request ID this response is for */
    request_id: string;

    /** Client ID */
    client_id: string;

    /** Error flag */
    error?: boolean;

    /** Error message (if error=true) */
    message?: string;
}

// ============================================================================
// Specific Request Types
// ============================================================================

export interface GetWorkflowRequest extends UIAPIRequest {
    endpoint: '/uiapi/get_workflow';
}

export interface GetWorkflowApiRequest extends UIAPIRequest {
    endpoint: '/uiapi/get_workflow_api';
}

export interface GetFieldRequest extends UIAPIRequest {
    endpoint: '/uiapi/get_field' | '/uiapi/get_fields';
    data: {
        /** Array of field paths to retrieve (e.g., ["prompt.text", "cfg"]) */
        fields: string[];
    };
}

export interface SetFieldsRequest extends UIAPIRequest {
    endpoint: '/uiapi/set_fields';
    data: {
        /** Single field to set (deprecated, use fields instead) */
        field?: [string, any];
        /** Array of [path, value] tuples */
        fields?: [string, any][];
    };
}

export interface SetConnectionRequest extends UIAPIRequest {
    endpoint: '/uiapi/set_connection';
    data: {
        /** Tuple of [source_path, target_path] */
        field: [string, string];
    };
}

export interface ExecuteRequest extends UIAPIRequest {
    endpoint: '/uiapi/execute';
}

export interface QueryFieldsRequest extends UIAPIRequest {
    endpoint: '/uiapi/query_fields';
}

export interface GetModelUrlRequest extends UIAPIRequest {
    endpoint: '/uiapi/get_model_url';
    data: {
        /** Single model checkpoint name (deprecated) */
        ckpt_name?: string;
        /** Array of requested checkpoint names */
        requested_ckpts?: string[];
        /** Array of already-existing checkpoints (to skip) */
        existing_ckpts?: string[];
    };
}

export interface ShowWorkflowDialogRequest extends UIAPIRequest {
    endpoint: '/uiapi/show_workflow_dialog';
    data: {
        /** Workflow data to display */
        workflow: any;
        /** Message to show user */
        message: string;
        /** Dialog title */
        title: string;
    };
}

export interface LoadWorkflowRequest extends UIAPIRequest {
    endpoint: '/uiapi/load_workflow';
    data: {
        /** Workflow data to load */
        workflow: any;
    };
}

// ============================================================================
// Specific Response Types
// ============================================================================

export interface GetWorkflowResponse extends UIAPIResponse {
    response: {
        workflow: any;
    };
}

export interface GetWorkflowApiResponse extends UIAPIResponse {
    response: {
        output: Record<string, any>;
    };
}

export interface GetFieldsResponse extends UIAPIResponse {
    response: Record<string, any>;
}

export interface QueryFieldsResponse extends UIAPIResponse {
    response: {
        nodes: NodeInfo[];
    };
}

export interface NodeInfo {
    id: number;
    type: string;
    title: string;
    inputs: string[];
    outputs: string[];
}

export interface GetModelUrlResponse extends UIAPIResponse {
    response: {
        /** Single URL (deprecated) */
        url?: string;
        /** Map of model names to URLs */
        [modelName: string]: string | undefined;
    } | null;
}

export interface WorkflowDialogResponse extends UIAPIResponse {
    response: {
        accepted: boolean;
        message: string;
    };
}

export interface LoadWorkflowResponse extends UIAPIResponse {
    response: {
        loaded: boolean;
        message: string;
    };
}

// ============================================================================
// Connection Manager Types
// ============================================================================

/**
 * Exponential backoff state for reconnection attempts
 */
export interface RetryState {
    /** Base retry interval (1 second) */
    baseInterval: number;

    /** Current retry interval (exponentially increasing) */
    currentInterval: number;

    /** Maximum retry interval cap (30 seconds) */
    maxInterval: number;

    /** Maximum number of consecutive retries before giving up */
    maxRetries: number;

    /** Current count of consecutive failures */
    consecutiveFailures: number;

    /** Timestamp of last logged retry (for throttling) */
    lastLoggedRetry: number;

    /** Minimum time between retry log messages (5 seconds) */
    logThrottle: number;

    /** True when max retries exhausted */
    gaveUp: boolean;
}

/**
 * Browser information sent to server on connection
 */
export interface BrowserInfo {
    /** Full user agent string */
    userAgent: string;

    /** Detected browser name */
    browser: 'Firefox' | 'Chrome' | 'Safari' | 'Edge' | 'Unknown';

    /** Platform string */
    platform: string;
}

/**
 * Connection status response from server
 */
export interface ConnectionStatusResponse {
    /** Whether WebUI websocket is connected */
    webui_connected: boolean;

    /** Server timestamp */
    timestamp?: number;
}

/**
 * WebUI ready request body
 */
export interface WebuiReadyRequest {
    browserInfo: BrowserInfo;
    client_id: string;
}

/**
 * WebUI ready response
 */
export interface WebuiReadyResponse {
    /** Client ID assigned by server */
    client_id: string;

    /** Success status */
    status: 'ok';
}

// ============================================================================
// Download Manager Types
// ============================================================================

/**
 * Download table structure mapping model names to download definitions
 */
export interface DownloadTable {
    [modelName: string]: ModelDefinition;
}

/**
 * Model definition for downloading
 */
export interface ModelDefinition {
    /** Download URL */
    url: string;

    /** Target path/filename */
    path?: string;

    /** Additional metadata */
    [key: string]: any;
}

/**
 * Download status response from server
 */
export interface DownloadStatusResponse {
    /** Overall download task status */
    status: 'pending' | 'downloading' | 'complete' | 'error';

    /** Per-model progress information */
    progress?: Record<string, DownloadProgress>;

    /** Download table (models being downloaded) */
    download_table?: DownloadTable;

    /** Error message (if status='error') */
    error?: string;
}

/**
 * Progress information for a single model download
 */
export interface DownloadProgress {
    /** Download status for this model */
    status: 'pending' | 'downloading' | 'complete' | 'error' | 'waiting_for_url';

    /** Bytes downloaded */
    downloaded?: number;

    /** Total bytes (if known) */
    total?: number;

    /** Download speed (bytes/sec) */
    speed?: number;

    /** Error message (if status='error') */
    error?: string;
}

/**
 * Add model URL request
 */
export interface AddModelUrlRequest {
    model_name: string;
    url: string;
}

/**
 * Add model URL response
 */
export interface AddModelUrlResponse {
    status: 'ok' | 'error';
    model_def?: ModelDefinition;
    message?: string;
}

// ============================================================================
// Node Utility Types
// ============================================================================

/**
 * Result from getNodeDataByPath() lookup
 */
export interface NodeData {
    /** Found node (or null) */
    node: LGraphNode | null;

    /** Found widget (or null) */
    widget: LGraphWidget | null;

    /** Found input slots (or indices if slotsAsIndices=true) */
    inputs: (LGraphNodeInput | number)[];

    /** Found output slots (or indices if slotsAsIndices=true) */
    outputs: (LGraphNodeOutput | number)[];
}

/**
 * Primary widget mapping: node type -> default widget name
 */
export interface PrimaryWidgets {
    CLIPSetLastLayer: 'stop_at_clip_layer';
    CLIPTextEncode: 'text';
    VAELoader: 'vae_name';
    TomePatchModel: 'ratio';
    SaveImage: 'filename_prefix';
    LoadImage: 'image';
    [nodeType: string]: string;
}

// ============================================================================
// Dialog Component Types
// ============================================================================

/**
 * Download dialog for single model URL input
 */
export interface DownloadDialog {
    modelName: string;
    modal: HTMLDivElement | null;
    content: HTMLDivElement | null;
    input: HTMLInputElement | null;

    show(): Promise<string | null>;
}

/**
 * Batch download dialog for multiple model URLs
 */
export interface BatchDownloadDialog {
    modelNames: string[];
    modal: HTMLDivElement | null;
    content: HTMLDivElement | null;
    inputs: Map<string, HTMLInputElement>;

    show(): Promise<Map<string, string> | null>;
}

/**
 * Workflow approval dialog
 */
export interface WorkflowDialog {
    workflow: any;
    message: string;
    title: string;
    modal: HTMLDivElement | null;
    content: HTMLDivElement | null;

    show(): Promise<boolean>;
}

// ============================================================================
// Constants
// ============================================================================

export const HEARTBEAT_INTERVAL: 5000;

export const RENAME_DEFAULTS: string[];

export const NODE_COLORS: Record<string, [number, number, number]>;

// ============================================================================
// Event Types
// ============================================================================

/**
 * Custom event detail for UIAPI requests
 */
export interface UIAPIEventDetail extends UIAPIRequest {
    // Inherits all fields from UIAPIRequest
}

/**
 * Custom event for UIAPI requests
 */
export interface UIAPIEvent extends CustomEvent<UIAPIEventDetail> {
    detail: UIAPIEventDetail;
}

// ============================================================================
// Service Exports
// ============================================================================

/**
 * Connection manager singleton
 */
export interface ConnectionManager {
    heartbeatInterval: number;
    intervalId: NodeJS.Timeout | null;
    isConnected: boolean;
    serverReachable: boolean;
    hasConnectedBefore: boolean;
    reconnectCallbacks: (() => Promise<void>)[];
    pendingRequests: Map<string, number>;
    retryState: RetryState;
    browserInfo?: BrowserInfo;

    /**
     * Initialize connection management
     */
    initialize(browserInfo: BrowserInfo): Promise<void>;

    /**
     * Register callback to fire on reconnection
     */
    onReconnect(callback: () => Promise<void>): void;

    /**
     * Track a pending request
     */
    trackPendingRequest(requestId: string): void;

    /**
     * Clear a pending request
     */
    clearPendingRequest(requestId: string): void;
}

/**
 * Download manager singleton
 */
export interface DownloadManager {
    checkInterval: number;
    activeChecks: Set<string>;

    /**
     * Start downloading models
     */
    downloadModels(downloadTable: DownloadTable): Promise<Response>;

    /**
     * Check download status (polling loop)
     */
    checkDownloadStatus(taskId: string): Promise<void>;

    /**
     * Stop checking a specific task
     */
    stopChecking(taskId: string): void;

    /**
     * Stop all active checks
     */
    stopAllChecks(): void;
}

// ============================================================================
// Exported singletons
// ============================================================================

export const connectionManager: ConnectionManager;
export const downloadManager: DownloadManager;

// ============================================================================
// API Handler Functions
// ============================================================================

export function setClientId(id: string): void;
export function getClientId(): string;
export function registerApiHandlers(): void;

export function handleGetWorkflow(event: UIAPIEvent): Promise<void>;
export function handleGetWorkflowApi(event: UIAPIEvent): Promise<void>;
export function handleGetFields(event: UIAPIEvent): Promise<void>;
export function handleSetFields(event: UIAPIEvent): Promise<void>;
export function handleSetConnection(event: UIAPIEvent): Promise<void>;
export function handleExecute(event: UIAPIEvent): Promise<void>;
export function handleQueryFields(event: UIAPIEvent): Promise<void>;
export function handleGetModelUrl(event: UIAPIEvent): Promise<void>;
export function handleShowWorkflowDialog(event: UIAPIEvent): Promise<void>;
export function handleLoadWorkflow(event: UIAPIEvent): Promise<void>;

// ============================================================================
// Node Utility Functions
// ============================================================================

export function getNodes(): LGraphNode[];
export function distance(x1: number, y1: number, x2: number, y2: number): number;
export function connectNodes(path1: string, path2: string): void;
export function selectNodeAt(x: number, y: number): void;
export function getNodeDataByPath(searchPath: string, slotsAsIndices?: boolean): NodeData;
export function simulateClick(x: number, y: number): void;

// ============================================================================
// Dialog Functions
// ============================================================================

export function showDownloadUrlDialog(modelName: string): Promise<string | null>;
export function showBatchDownloadUrlDialog(modelNames: string[]): Promise<Map<string, string> | null>;
export function showWorkflowDialog(workflow: any, message: string, title: string): Promise<boolean>;
