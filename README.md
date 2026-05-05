# comfy-ctrl

Remote control bridge for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Drive workflows programmatically from any Python program — or let an LLM agent do it through the built-in MCP server.

Three layers in one package:

| Layer | What it does |
|-------|-------------|
| **Server extension** | Registers `/uiapi/*` HTTP routes on ComfyUI's PromptServer, relays commands to the browser frontend via WebSocket |
| **Python client** | `ComfyClient` — sync and async API for connecting to ComfyUI, manipulating fields, executing workflows, downloading models |
| **MCP server** | Exposes ComfyUI image generation as tools for Claude Code, Cursor, and other MCP-compatible agents |

## How it works

```
Your program ──HTTP──▶ ComfyUI server (comfy-ctrl routes) ──WS──▶ Browser frontend
                                                           ◀──WS── (response relayed back)
```

comfy-ctrl is a relay: it forwards commands from your program to the ComfyUI web UI running in a browser tab, then returns the response. This lets you manipulate the live workflow without exporting/importing JSON.

## Install

### As a ComfyUI extension

Clone into your ComfyUI `custom_nodes/` directory:

```bash
git clone https://github.com/holo-q/comfy-ctrl custom_nodes/comfy-ctrl
```

### MCP server

```bash
uvx --from /path/to/comfy-ctrl comfy-ctrl-mcp
```

Or install into Claude Code / Claude Desktop via the one-click installer in the ComfyUI web UI (Settings → comfy-ctrl → Install MCP).

## Client usage

```python
from comfy_client import ComfyClient

client = ComfyClient("http://localhost:8188")
client.connect_webui()

# Set fields by node_name.widget_name path
client.set("ksampler.seed", 42)
client.set("prompt.text", "a cat in space")

# Wire nodes together
client.connect("model_loader.MODEL", "ksampler.model")

# Execute and get output images (numpy arrays)
images = client.execute()
```

### Async

Every method has an async variant:

```python
await client.set_async("prompt.text", "a cat in space")
images = await client.execute_async()
```

### Model downloads

Orchestrate model downloads on a remote ComfyUI server:

```python
from model_defs import ModelDef, LoraDef

client.download_models([
    ModelDef("sd_xl_base_1.0.safetensors", civitai="https://civitai.com/models/..."),
    LoraDef("detail_tweaker_xl.safetensors", huggingface="org/repo/file.safetensors"),
])
```

Supports CivitAI, HuggingFace, and Google Drive sources.

## MCP tools

When running as an MCP server, comfy-ctrl exposes:

- **generate_image** — auto-discovers installed API providers (Gemini, OpenAI, Flux, Stability, etc.) and builds workflows dynamically
- **execute** — run the current browser-loaded workflow
- **execute_workflow** — run arbitrary API-format workflows with field overrides

## Project structure

```
comfy_client.py     Python client library (ComfyClient)
uiapi.py            Server extension — HTTP routes, WebSocket relay
mcp_server.py       MCP server wrapping the client for LLM agents
model_defs.py       Model definition dataclasses + download orchestration
civitai.py          CivitAI API client
__init__.py         ComfyUI plugin entry point
js/                 Browser-side frontend extension
  services/         WebSocket connection, API handlers, download manager
  components/       UI dialogs (download, workflow, MCP install)
  utils/            Node graph utilities
```

## License

MIT
