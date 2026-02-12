import { app } from "../../scripts/app.js";
import { connectionManager } from "./services/ConnectionManager.js";
import { registerApiHandlers } from "./services/ApiHandlers.js";
import { selectNodeAt } from "./utils/nodeUtils.js";
import { RENAME_DEFAULTS } from "./config/constants.js";
import { showMCPInstallDialog } from "./components/MCPInstallDialog.js";

let lastClickTime = null;
let lastRenameValue = "";

/**
 * Handle mouse events
 * @param {MouseEvent} e - Mouse event
 */
function handleMouseEvent(e) {
    console.debug(`[UIAPI] 🖱️ Mouse event: ${e.type}`);
    
    if (e.type === "mousedown" || e.type === "pointerdown") {
        console.debug("[UIAPI] 🖱️ Primary interaction detected");
    }

    lastClickTime = Date.now();
}

/**
 * Handle keyboard events
 * @param {KeyboardEvent} e - Keyboard event
 * @returns {boolean} Whether the event was handled
 */
function handleKeyEvent(e) {
    if (e.type !== "keydown" || e.repeat) return false;

    console.debug(`[UIAPI] ⌨️ Key pressed: ${e.key}`);
    
    if (e.key === 'Escape') {
        console.debug("[UIAPI] ⌨️ ESC - Closing menus");
        app.canvas.search_box?.close();
        return true;
    }

    const canvas = app.canvas;
    const movedist = Math.hypot(canvas.mouse[0] - canvas.last_mouse[0], 
                               canvas.mouse[1] - canvas.last_mouse[1]);
    const auto = movedist < 10;

    // F2 rename feature
    if (e.key === 'F2') {
        if (auto) {
            // Auto-select next numbered rename_default if last_rename_value ends with a number
            const lastChar = lastRenameValue.slice(-1);
            const lastNum = parseInt(lastChar);
            if (!isNaN(lastNum)) {
                const nextNum = (lastNum + 1) % 10;
                const nextValue = lastRenameValue.slice(0, -1) + nextNum;
                if (RENAME_DEFAULTS.includes(nextValue)) {
                    console.log("[UIAPI] 🏷️ Auto-incrementing node name to:", nextValue);
                    for (let node of Object.values(canvas.selected_nodes)) {
                        node.title = nextValue;
                    }
                    lastRenameValue = nextValue;
                }
            }
        } else {
            console.log("[UIAPI] 🏷️ Opening rename menu for node");
            selectNodeAt(canvas.graph_mouse[0], canvas.graph_mouse[1]);
            showRenameContextMenu(e);
        }
        return true;
    }

    return false;
}

/**
 * Show the rename context menu
 * @param {Event} e - Event that triggered the menu
 * @returns {boolean} Whether the menu was shown
 */
function showRenameContextMenu(e) {
    const canvas = app.canvas;
    const selectedNode = Object.values(canvas.selected_nodes)[0];
    if (!selectedNode) return false;

    console.log("[UIAPI] 📝 Opening rename context menu");
    
    const menu = new LiteGraph.ContextMenu(RENAME_DEFAULTS, {
        title: `Rename '${selectedNode.title}' ...`,
        left: canvas.mouse[0] - 30,
        top: canvas.mouse[1] - 30,
        callback: (value) => {
            console.log("[UIAPI] 🏷️ Renaming node to:", value);
            for (let node of Object.values(canvas.selected_nodes)) {
                node.title = value;
            }
            lastRenameValue = value;
        }
    });

    return false;
}

// Register the extension
app.registerExtension({
    name: "uiapi",
    version: "1.0",
    async setup() {
        console.log("[UIAPI] ==== Setup Starting ====");
        
        // Get browser info
        const browserInfo = {
            userAgent: navigator.userAgent,
            browser: (function() {
                const ua = navigator.userAgent;
                if (ua.includes('Firefox')) return 'Firefox';
                if (ua.includes('Chrome')) return 'Chrome';
                if (ua.includes('Safari')) return 'Safari';
                if (ua.includes('Edge')) return 'Edge';
                return 'Unknown';
            })(),
            platform: navigator.platform
        };

        // Register API event handlers
        registerApiHandlers();

        // Initialize connection manager with browser info
        await connectionManager.initialize(browserInfo);

        // Set up mouse event handlers
        const canvas = app.canvas;
        const originalHandlers = {
            onMouseDown: canvas.onMouseDown,
            onMouseUp: canvas.onMouseUp,
            onMouseMove: canvas.onMouseMove
        };

        canvas.onMouseDown = e => {
            handleMouseEvent(e);
            if (originalHandlers.onMouseDown) originalHandlers.onMouseDown(e);
        };

        canvas.onMouseUp = e => {
            handleMouseEvent(e);
            if (originalHandlers.onMouseUp) originalHandlers.onMouseUp(e);
        };

        canvas.onMouseMove = e => {
            handleMouseEvent(e);
            if (originalHandlers.onMouseMove) originalHandlers.onMouseMove(e);
        };

        // Set up keyboard event handlers
        window.addEventListener("keydown", e => {
            if (handleKeyEvent(e)) {
                e.preventDefault();
                e.stopImmediatePropagation();
            }
        });

        window.addEventListener("keyup", handleKeyEvent);

        // MCP Install button — one-click config for Claude Desktop/Code
        const mcpButton = document.createElement('button');
        mcpButton.textContent = 'MCP';
        mcpButton.title = 'Install MCP server for Claude Desktop / Claude Code';
        mcpButton.style.cssText = `
            position: fixed;
            bottom: 10px;
            right: 10px;
            z-index: 9999;
            background: rgba(45, 45, 45, 0.9);
            border: 1px solid #555;
            color: #ccc;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: bold;
            backdrop-filter: blur(4px);
            transition: all 0.2s;
        `;
        mcpButton.onmouseover = () => {
            mcpButton.style.borderColor = '#2ecc71';
            mcpButton.style.color = '#fff';
            mcpButton.style.background = 'rgba(46, 204, 113, 0.15)';
        };
        mcpButton.onmouseout = () => {
            mcpButton.style.borderColor = '#555';
            mcpButton.style.color = '#ccc';
            mcpButton.style.background = 'rgba(45, 45, 45, 0.9)';
        };
        mcpButton.onclick = () => showMCPInstallDialog();
        document.body.appendChild(mcpButton);

        console.log("[UIAPI] ✓ Setup completed successfully");
    }
});
