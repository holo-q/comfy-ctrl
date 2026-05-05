/**
 * TypeScript type definitions for ComfyUI and LiteGraph external dependencies
 * These types describe the ComfyUI app API and LiteGraph library interfaces
 * that the UIAPI extension depends on.
 */

// ============================================================================
// ComfyUI App API Types
// ============================================================================

export interface ComfyApp {
    /** The LiteGraph canvas instance */
    canvas: LGraphCanvas;

    /** The current graph being edited */
    graph: LGraph;

    /** Connection status for UIAPI bridge */
    uiapi_connected?: boolean;

    /**
     * Register an extension with ComfyUI
     * @param extension Extension configuration object
     */
    registerExtension(extension: ComfyExtension): void;

    /**
     * Convert the current graph to a prompt for execution
     * @returns Promise resolving to the prompt data
     */
    graphToPrompt(): Promise<PromptData>;

    /**
     * Load graph data into the canvas
     * @param graphData Graph data to load (supports API format with auto-layout)
     */
    loadGraphData(graphData: any): Promise<void>;
}

export interface ComfyExtension {
    /** Unique name for the extension */
    name: string;

    /** Version string */
    version?: string;

    /** Setup function called when extension is loaded */
    setup?(): Promise<void> | void;

    /** Additional lifecycle hooks */
    [key: string]: any;
}

export interface PromptData {
    /** The workflow structure */
    workflow?: any;

    /** API-formatted output for execution */
    output?: Record<string, any>;

    /** Additional prompt data fields */
    [key: string]: any;
}

// ============================================================================
// ComfyUI API Types
// ============================================================================

export interface ComfyApi {
    /**
     * Fetch from the ComfyUI API
     * @param route API route
     * @param options Fetch options
     */
    fetchApi(route: string, options?: RequestInit): Promise<Response>;

    /**
     * Queue a prompt for execution
     * @param number Queue position (0 for immediate)
     * @param prompt Prompt data
     */
    queuePrompt(number: number, prompt: PromptData): Promise<any>;

    /**
     * Add event listener for API events
     * @param event Event name (e.g., "/uiapi/get_workflow")
     * @param callback Event handler
     */
    addEventListener(event: string, callback: (event: CustomEvent) => void): void;

    /**
     * Remove event listener
     * @param event Event name
     * @param callback Event handler to remove
     */
    removeEventListener(event: string, callback: (event: CustomEvent) => void): void;
}

// ============================================================================
// LiteGraph Types
// ============================================================================

export interface LGraph {
    /** Internal nodes array */
    _nodes: LGraphNode[];

    /**
     * Get node at a specific position
     * @param x X coordinate
     * @param y Y coordinate
     * @param nodes Array of nodes to search (optional)
     * @param margin Margin for hit detection
     */
    getNodeOnPos(x: number, y: number, nodes?: LGraphNode[], margin?: number): LGraphNode | null;

    /** Clear all nodes from the graph */
    clear(): void;
}

export interface LGraphCanvas {
    /** Mouse coordinates in canvas space */
    mouse: [number, number];

    /** Last mouse coordinates */
    last_mouse: [number, number];

    /** Mouse coordinates in graph space */
    graph_mouse: [number, number];

    /** Currently selected nodes */
    selected_nodes: Record<string, LGraphNode>;

    /** Currently visible nodes */
    visible_nodes: LGraphNode[];

    /** Search box widget (if open) */
    search_box?: {
        close(): void;
    };

    /** Original mouse event handlers (for wrapping) */
    onMouseDown?: (e: MouseEvent) => void;
    onMouseUp?: (e: MouseEvent) => void;
    onMouseMove?: (e: MouseEvent) => void;

    /**
     * Deselect all nodes
     */
    deselectAllNodes(): void;

    /**
     * Select specific nodes
     * @param nodes Array of nodes to select
     */
    selectNodes(nodes: LGraphNode[]): void;

    /**
     * Mark canvas as dirty to trigger redraw
     * @param background Whether to redraw background (node bodies, connections)
     * @param foreground Whether to redraw foreground (widgets, selections)
     */
    setDirtyCanvas(background: boolean, foreground: boolean): void;
}

export interface LGraphNode {
    /** Node ID (unique within graph) */
    id: number;

    /** Node type (e.g., "CLIPTextEncode", "KSampler") */
    type: string;

    /** Node title (user-editable label) */
    title: string;

    /** Node position */
    pos: [number, number];

    /** Node size */
    size: [number, number];

    /** Node input slots */
    inputs?: LGraphNodeInput[];

    /** Node output slots */
    outputs?: LGraphNodeOutput[];

    /** Node widgets (UI controls) */
    widgets?: LGraphWidget[];

    /**
     * Connect this node's output to another node's input
     * @param outputSlot Output slot index or object
     * @param targetNode Target node
     * @param inputSlot Input slot index or object
     */
    connect(outputSlot: number | LGraphNodeOutput, targetNode: LGraphNode, inputSlot: number | LGraphNodeInput): void;
}

export interface LGraphNodeInput {
    /** Input name */
    name: string;

    /** Input type (e.g., "CLIP", "IMAGE") */
    type: string;

    /** Link ID if connected */
    link?: number;
}

export interface LGraphNodeOutput {
    /** Output name */
    name: string;

    /** Output type (e.g., "CLIP", "IMAGE") */
    type: string;

    /** Array of link IDs if connected */
    links?: number[];
}

export interface LGraphWidget {
    /** Widget name (matches node parameter) */
    name: string;

    /** Widget type (e.g., "text", "number", "combo") */
    type: string;

    /** Widget value */
    value: any;

    /** Widget options (for combo boxes, etc.) */
    options?: {
        values?: string[];
        [key: string]: any;
    };

    /**
     * Optional callback invoked when widget value changes
     * @param value New value
     * @param canvas The graph canvas
     * @param node The node containing this widget
     * @param pos Mouse position (may be null)
     * @param event Original event (may be null)
     */
    callback?: (value: any, canvas: LGraphCanvas, node: LGraphNode, pos: any, event: any) => void;
}

// ============================================================================
// LiteGraph Context Menu Types
// ============================================================================

export interface LiteGraphContextMenuOptions {
    /** Menu title */
    title?: string;

    /** Left position (pixels) */
    left?: number;

    /** Top position (pixels) */
    top?: number;

    /** Callback when item is selected */
    callback?: (value: any, options: any, event: MouseEvent, parentMenu: any, node: LGraphNode) => void;

    /** Event that triggered the menu */
    event?: MouseEvent;

    /** Parent menu (for submenus) */
    parentMenu?: any;

    /** Node associated with the menu */
    node?: LGraphNode;
}

export class LiteGraphContextMenu {
    /**
     * Create a context menu
     * @param items Array of menu items or strings
     * @param options Menu configuration options
     */
    constructor(items: (string | ContextMenuItem)[], options?: LiteGraphContextMenuOptions);

    /** Close the menu */
    close(): void;
}

export interface ContextMenuItem {
    /** Item label */
    content?: string;

    /** Item value */
    value?: any;

    /** Item callback */
    callback?: (value: any, options: any, event: MouseEvent, parentMenu: any, node: LGraphNode) => void;

    /** Whether item is disabled */
    disabled?: boolean;

    /** Submenu items */
    submenu?: ContextMenuItem[];
}

// ============================================================================
// Global LiteGraph namespace
// ============================================================================

declare global {
    const LiteGraph: {
        ContextMenu: typeof LiteGraphContextMenu;
        [key: string]: any;
    };

    const app: ComfyApp;
    const api: ComfyApi;
}

// ============================================================================
// Module exports for ES modules
// ============================================================================

export const app: ComfyApp;
export const api: ComfyApi;
