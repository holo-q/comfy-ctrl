/**
 * Promise-based modal dialog for one-click MCP server installation.
 * Follows the same pattern as WorkflowDialog.js: dark theme, inline styles,
 * promise-based show() that resolves on close/escape.
 *
 * Fetches MCP status from the backend, renders target checkboxes (Claude Desktop,
 * Claude Code), and POSTs selected targets for installation.
 */

const TARGET_LABELS = {
    claude_desktop: "Claude Desktop",
    claude_code: "Claude Code",
};

export class MCPInstallDialog {
    constructor() {
        this.modal = null;
        this.content = null;
        this.statusData = null;
        // References to per-target DOM elements for post-install updates
        this.targetRows = {};
    }

    /**
     * Fetch MCP status, build the modal, attach to DOM, and return a promise
     * that resolves when the user closes the dialog.
     * @returns {Promise<void>}
     */
    show() {
        return new Promise(async (resolve) => {
            this._resolve = resolve;

            try {
                const resp = await fetch("/uiapi/mcp_status");
                if (!resp.ok) throw new Error(`Status ${resp.status}`);
                this.statusData = await resp.json();
            } catch (err) {
                console.error("[MCPInstallDialog] Failed to fetch MCP status:", err);
                // Show a minimal error dialog so the user isn't left hanging
                this.statusData = null;
            }

            this.createModal();
            document.body.appendChild(this.modal);

            // Escape key dismisses
            this._handleEscape = (e) => {
                if (e.key === "Escape") this.close();
            };
            document.addEventListener("keydown", this._handleEscape);
        });
    }

    /**
     * Remove modal from DOM, clean up listeners, resolve the promise.
     * @private
     */
    close() {
        if (this.modal) this.modal.remove();
        document.removeEventListener("keydown", this._handleEscape);
        if (this._resolve) this._resolve();
    }

    // ── DOM Construction ────────────────────────────────────────────────

    /**
     * Build the full modal DOM tree.
     * @private
     */
    createModal() {
        // Overlay
        this.modal = document.createElement("div");
        this.modal.style.cssText = `
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 99999;
        `;

        // Content card
        this.content = document.createElement("div");
        this.content.style.cssText = `
            background: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            min-width: 460px;
            max-width: 560px;
            color: #ffffff;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
        `;

        // Title
        const title = document.createElement("h3");
        title.textContent = "Install MCP Server for Claude";
        title.style.cssText = `
            margin: 0 0 8px 0;
            font-size: 18px;
            color: #ffffff;
        `;
        this.content.appendChild(title);

        // Subtitle
        const subtitle = document.createElement("p");
        subtitle.textContent =
            "Configure Claude Desktop and Claude Code to control ComfyUI via MCP tools.";
        subtitle.style.cssText = `
            margin: 0 0 14px 0;
            font-size: 14px;
            color: #aaaaaa;
        `;
        this.content.appendChild(subtitle);

        if (!this.statusData) {
            // Error state -- couldn't reach backend
            const errMsg = document.createElement("p");
            errMsg.textContent = "Failed to fetch MCP status from the server.";
            errMsg.style.cssText = "color: #e74c3c; margin: 20px 0;";
            this.content.appendChild(errMsg);

            const buttons = this.createButtonRow(null);
            this.content.appendChild(buttons);
        } else {
            // Plugin path
            if (this.statusData.plugin_path) {
                const pathLine = document.createElement("p");
                pathLine.textContent = this.statusData.plugin_path;
                pathLine.style.cssText = `
                    margin: 0 0 16px 0;
                    font-family: monospace;
                    font-size: 12px;
                    color: #888888;
                `;
                this.content.appendChild(pathLine);
            }

            // Target rows container
            const targetsContainer = document.createElement("div");
            targetsContainer.style.cssText = "margin: 0 0 4px 0;";

            const targets = this.statusData.targets || {};
            for (const [key, info] of Object.entries(targets)) {
                const row = this.createTargetRow(key, info);
                targetsContainer.appendChild(row);
            }
            this.content.appendChild(targetsContainer);

            // Buttons
            const buttons = this.createButtonRow(targets);
            this.content.appendChild(buttons);
        }

        this.modal.appendChild(this.content);
    }

    /**
     * Build a single target row with checkbox, label, config path, and status badge.
     * @private
     */
    createTargetRow(key, info) {
        const row = document.createElement("div");
        row.style.cssText = `
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px;
            border-bottom: 1px solid #444;
        `;

        // Checkbox
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.dataset.targetKey = key;
        checkbox.style.cssText = `
            accent-color: #2ecc71;
            width: 16px; height: 16px;
            cursor: pointer;
        `;
        // Pre-check if NOT already installed
        checkbox.checked = !info.installed;
        row.appendChild(checkbox);

        // Label block
        const labelDiv = document.createElement("div");
        labelDiv.style.cssText = "flex: 1;";

        const nameSpan = document.createElement("div");
        nameSpan.textContent = TARGET_LABELS[key] || key;
        nameSpan.style.cssText = "font-weight: bold; font-size: 14px;";
        labelDiv.appendChild(nameSpan);

        if (info.path) {
            const pathSpan = document.createElement("div");
            pathSpan.textContent = info.path;
            pathSpan.style.cssText = `
                font-family: monospace;
                font-size: 11px;
                color: #888888;
                margin-top: 2px;
            `;
            labelDiv.appendChild(pathSpan);
        }

        row.appendChild(labelDiv);

        // Status badge
        const badge = document.createElement("span");
        badge.style.cssText = "font-size: 12px; white-space: nowrap;";
        if (info.installed) {
            badge.textContent = "Installed";
            badge.style.color = "#2ecc71";
        } else if (info.config_exists) {
            badge.textContent = "Not installed";
            badge.style.color = "#999999";
        } else {
            badge.textContent = "Will create";
            badge.style.color = "#f39c12";
        }
        row.appendChild(badge);

        // Stash references for post-install updates
        this.targetRows[key] = { checkbox, badge, row };

        return row;
    }

    /**
     * Build the button row (Close + Install).
     * @private
     * @param {Object|null} targets - If null, only show Close button (error state).
     */
    createButtonRow(targets) {
        const wrapper = document.createElement("div");
        wrapper.style.cssText = `
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-top: 20px;
        `;

        const closeBtn = this.createButton("Close", "#555555");
        closeBtn.onclick = () => this.close();
        wrapper.appendChild(closeBtn);

        if (targets) {
            this.installBtn = this.createButton("Install", "#2ecc71");
            this.installBtn.onclick = () => this.doInstall();
            wrapper.appendChild(this.installBtn);
        }

        return wrapper;
    }

    // ── Actions ─────────────────────────────────────────────────────────

    /**
     * Collect checked targets, POST to /uiapi/install_mcp, and update badges
     * with per-target results.
     * @private
     */
    async doInstall() {
        const selectedKeys = Object.entries(this.targetRows)
            .filter(([_, refs]) => refs.checkbox.checked)
            .map(([key]) => key);

        if (selectedKeys.length === 0) return;

        // Disable button during install
        this.installBtn.disabled = true;
        this.installBtn.textContent = "Installing...";
        this.installBtn.style.opacity = "0.6";
        this.installBtn.style.cursor = "default";

        try {
            const resp = await fetch("/uiapi/install_mcp", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ targets: selectedKeys }),
            });

            if (!resp.ok) throw new Error(`Status ${resp.status}`);
            const data = await resp.json();

            // Update each target badge based on backend response
            for (const [key, result] of Object.entries(data.results || {})) {
                const refs = this.targetRows[key];
                if (!refs) continue;

                const status = result.status || result;
                const badge = refs.badge;

                if (status === "installed" || status === "updated") {
                    badge.textContent = status === "updated" ? "Updated" : "Installed";
                    badge.style.color = "#2ecc71";
                    refs.checkbox.checked = false;
                } else if (status === "already_installed") {
                    badge.textContent = "Already installed";
                    badge.style.color = "#2ecc71";
                    refs.checkbox.checked = false;
                } else if (status === "error") {
                    const msg = result.message || result.error || "Unknown error";
                    badge.textContent = `Error: ${msg}`;
                    badge.style.color = "#e74c3c";
                }
            }
        } catch (err) {
            console.error("[MCPInstallDialog] Install failed:", err);
            // Surface the error visually on the button
            this.installBtn.textContent = "Error";
            this.installBtn.style.background = "#e74c3c";
            return;
        }

        // Done state
        this.installBtn.textContent = "Done";
        this.installBtn.style.opacity = "1";
        this.installBtn.style.cursor = "pointer";
        this.installBtn.disabled = false;
    }

    // ── Shared Helpers ──────────────────────────────────────────────────

    /**
     * Create a styled button -- same pattern as WorkflowDialog.createButton
     * @private
     */
    createButton(text, bgColor) {
        const button = document.createElement("button");
        button.textContent = text;
        button.style.cssText = `
            padding: 8px 16px; border: none; border-radius: 4px;
            background: ${bgColor}; color: white; cursor: pointer;
            font-size: 1em; transition: opacity 0.2s;
        `;
        button.onmouseover = () => (button.style.opacity = "0.8");
        button.onmouseout = () => (button.style.opacity = "1");
        return button;
    }
}

/**
 * Show the MCP install dialog and wait for the user to close it.
 * This is the primary entry point -- matches the showWorkflowDialog() pattern.
 * @returns {Promise<void>} Resolves when the dialog is dismissed.
 */
export async function showMCPInstallDialog() {
    const dialog = new MCPInstallDialog();
    return await dialog.show();
}
