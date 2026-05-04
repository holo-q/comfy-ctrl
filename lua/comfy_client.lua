-- comfy_client.lua

--------------------------------------------------------------------------------
-- Dependencies
--------------------------------------------------------------------------------
local http_request = require "http.request"
local websocket    = require "http.websocket"
local json         = require "lunajson"

--------------------------------------------------------------------------------
-- Logging helper (safely stringify tables)
--------------------------------------------------------------------------------
local function log(level, ...)
    local parts = {}
    for i = 1, select("#", ...) do
        local v = select(i, ...)
        if type(v) == "table" then
            local ok, s = pcall(json.encode, v)
            parts[#parts+1] = ok and s or tostring(v)
        else
            parts[#parts+1] = tostring(v)
        end
    end
    io.stderr:write(string.format("[%s] %s\n", level, table.concat(parts, " ")))
end

--------------------------------------------------------------------------------
-- UUID4 generator (RFC4122 v4)
--------------------------------------------------------------------------------
local function uuid4()
    local random = math.random
    local template = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx"
    return (template:gsub("[xy]", function(c)
        local v = (c == "x") and random(0, 0xf) or random(8, 0xb)
        return string.format("%x", v)
    end))
end

--------------------------------------------------------------------------------
-- ComfyClient class
--------------------------------------------------------------------------------
local ComfyClient = {}
ComfyClient.__index = ComfyClient

--- Create a new client instance.
-- @param server_address string, e.g. "127.0.0.1:8188"
-- @return instance
function ComfyClient:new(server_address)
    log("INFO", "Initializing ComfyClient with", server_address)
    local inst = setmetatable({}, self)
    inst.server_address = server_address or "127.0.0.1:8188"
    inst.client_id      = uuid4()
    inst._webui_ready   = false
    inst.ws             = nil
    return inst
end

--------------------------------------------------------------------------------
-- Internal sync HTTP request (JSON or PNG)
--------------------------------------------------------------------------------
-- @param method "GET" or "POST"
-- @param path   e.g. "/uiapi/execute" (or full URL)
-- @param data   Lua table for POST (ignored on GET)
-- @return JSON-decoded table, or raw bytes for /view
function ComfyClient:_make_request_once(method, path, data)
    log("DEBUG", "HTTP", method, path)

    -- Build the URI
    local uri = path:match("^https?://") and path
             or ("http://" .. self.server_address .. path)
    log("DEBUG", "Full URL:", uri)

    local req = http_request.new_from_uri(uri)
    req.headers:upsert(":method",    method)
    req.headers:upsert(":authority", self.server_address)
    req.headers:upsert(":scheme",    "http")
    req.headers:upsert(":path",      path)

    -- Only POST requests carry a JSON body
    data = data or {}
    data.client_id = self.client_id
    local body = json.encode(data)
    log("DEBUG", "Request body:", body)
    req:set_body(body)
    req.headers:upsert("content-type",   "application/json")
    req.headers:upsert("content-length", tostring(#body))

    local headers, stream, err = req:go()
    if not headers then
        error("HTTP request failed: " .. tostring(err))
    end

    local status = tonumber(headers:get(":status") or 0)
    log("DEBUG", "HTTP status:", status)
    if status < 200 or status >= 300 then
        local err_body = stream and stream:get_body_as_string() or ""
        error(string.format("HTTP error %d: %s", status, err_body))
    end

    local resp_body = stream:get_body_as_string()
    log("DEBUG", "Response length:", #resp_body)

    -- Return raw bytes for image fetch
    if path:find("/view") then
        return resp_body
    end

    -- Otherwise decode JSON
    local obj = json.decode(resp_body)
    if not obj then
        error("Failed to decode JSON response")
    end
    return obj
end

--------------------------------------------------------------------------------
-- Ensure HTTP + WebSocket connection
--------------------------------------------------------------------------------
-- @param retry boolean; if false (default), do not retry on first failure
-- @return true if connection ready, false if non-retryable failure
function ComfyClient:ensure_connection(retry)
    retry = retry == true  -- default false
    log("DEBUG", "ensure_connection(retry=" .. tostring(retry) .. ")")

    if self._webui_ready then
        log("INFO", "Already connected")
        return true
    end

    local backoff, max_backoff = 1, 30
    while true do
        -- 1) HTTP health check
        log("INFO", "Checking /uiapi/connection_status")
        local ok, resp_or_err = pcall(self._make_request_once, self, "GET", "/uiapi/connection_status")
        if not ok then
            log("WARN", "HTTP check failed:", resp_or_err)
            if not retry then return false end
        else
            log("DEBUG", "connection_status:", resp_or_err)
            if not resp_or_err.webui_connected then
                log("WARN", "WebUI not connected yet")
                if not retry then return false end
            else
                log("INFO", "WebUI HTTP OK, proceed to WebSocket")
                -- 2) WebSocket handshake
                local ws_url = string.format("ws://%s/ws?clientId=%s", self.server_address, self.client_id)
                log("INFO", "Connecting WebSocket to", ws_url)
                local ws, ws_err = websocket.new_from_uri(ws_url)
                if not ws then
                    log("ERROR", "WebSocket URI creation failed:", ws_err)
                    if not retry then return false end
                else
                    local ok2, conn_err = ws:connect()
                    if not ok2 then
                        log("ERROR", "WebSocket connect failed:", conn_err)
                        if not retry then return false end
                    else
                        log("INFO", "WebSocket connected")
                        self.ws = ws
                        self._webui_ready = true
                        return true
                    end
                end
            end
        end

        -- Sleep & backoff if retrying
        if not retry then
            log("INFO", "Not retrying, exiting ensure_connection")
            return false
        end

        log("INFO", string.format("Retrying in %d seconds…", backoff))
        os.execute("sleep " .. backoff)
        backoff = math.min(backoff * 2, max_backoff)
    end
end

--------------------------------------------------------------------------------
-- Public: JSON GET
--------------------------------------------------------------------------------
function ComfyClient:json_get(path)
    log("DEBUG", "json_get", path)
    return self:_make_request_once("GET", path)
end

--------------------------------------------------------------------------------
-- Public: JSON POST
--------------------------------------------------------------------------------
function ComfyClient:json_post(path, data)
    log("DEBUG", "json_post", path)
    return self:_make_request_once("POST", path, data)
end

--------------------------------------------------------------------------------
-- Public: Set a single field in the workflow
--------------------------------------------------------------------------------
function ComfyClient:set(path, value)
    log("INFO", "Setting", path, "→", value)
    return self:json_post("/uiapi/set_fields", {
        fields = { { path, value } }
    })
end

--------------------------------------------------------------------------------
-- Helper to find the SaveImage node
--------------------------------------------------------------------------------
local function find_output_node(tbl)
    for id, node in pairs(tbl) do
        if type(node) == "table" and (node.class_type == "SaveImage"
           or node.class_type == "Image Save") then
            return id
        end
        if type(node) == "table" then
            local sub = find_output_node(node)
            if sub then return sub end
        end
    end
end

--------------------------------------------------------------------------------
-- Public: Execute workflow and return PNG bytes
--------------------------------------------------------------------------------
function ComfyClient:execute()
    log("INFO", "Issuing execute")
    local resp = self:json_post("/uiapi/execute")
    log("DEBUG", "execute response:", resp)

    local pid = resp.prompt_id or error("execute: missing prompt_id")
    log("INFO", "Prompt ID:", pid)

    -- Wait for execution_success via WS
    local deadline = os.time() + 90
    while os.time() < deadline do
        log("DEBUG", "Waiting on WebSocket…")
        local ok, msg = pcall(self.ws.receive, self.ws)
        if ok and msg then
            log("DEBUG", "WS msg:", msg)
            local j = json.decode(msg)
            if j and j.type == "execution_success" then
                log("INFO", "Execution succeeded")
                break
            end
        end
        os.execute("sleep 1")
    end

    -- Locate output node
    log("INFO", "Fetching workflow for output node")
    local wf = self:json_get("/uiapi/get_workflow")
    local root = wf.response or wf
    local out_id = find_output_node(root)
    if not out_id then error("No output node found") end
    log("INFO", "Output node:", out_id)

    -- Get history & image info
    log("INFO", "Fetching history for", pid)
    local history = self:_make_request_once("GET", "/history/" .. pid)
    local info = history[pid].outputs[out_id].images[1]
    log("INFO", "Image info:", info)

    -- Download final PNG
    local qp = string.format("?filename=%s&subfolder=%s&type=%s",
                              info.filename, info.subfolder, info.type)
    log("INFO", "GET /view" .. qp)
    local img = self:_make_request_once("GET", "/view" .. qp)
    log("INFO", "Received image bytes:", #img)
    return img
end

--------------------------------------------------------------------------------
-- Module export
--------------------------------------------------------------------------------
return ComfyClient

