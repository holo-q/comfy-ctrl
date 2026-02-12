"""
MCP Server for ComfyUI-uiapi
=============================
Bridges LLM tool-use (Claude Code, Cursor, etc.) to ComfyUI's web frontend
via the uiapi extension. Wraps ComfyClient's async Python API as MCP tools.

Architecture:
  LLM (MCP client) <-stdio-> this server <-HTTP/WS-> ComfyUI + uiapi extension

CRITICAL: stdout is the MCP protocol wire — only valid JSON-RPC may appear there.
comfy_client, model_defs, and civitai create Rich Console() instances and call
logging.basicConfig(RichHandler()) at *import time*, both defaulting to stdout.
We avoid this by lazy-importing those modules inside the functions that need them,
so their import-time side effects never fire during MCP protocol initialization.
"""

import base64
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.utilities.types import Image as MCPImage

log = logging.getLogger("comfyui-mcp")

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "comfyui",
    instructions="Control ComfyUI image generation. Use generate_image as the "
    "primary tool — it builds and executes workflows in one call. Routes "
    "(gemini, diffusion) construct workflows dynamically; templates store "
    "reusable static workflows. Use list_routes to see what's available.",
)

# ---------------------------------------------------------------------------
# Lazy client — connects on first tool call
# ---------------------------------------------------------------------------
_client = None


async def get_client(require_uiapi=False):
    """Lazy-connect to ComfyUI. Reads COMFYUI_ADDRESS env var.

    By default does NOT require a WebUI browser tab — headless tools like
    generate_image post directly to /prompt. Pass require_uiapi=True for
    tools that manipulate the browser-loaded workflow (query_fields, etc.).
    """
    global _client
    if _client is not None:
        if require_uiapi:
            await _client.ensure_connection_async(require_uiapi=True)
        return _client

    from comfy_client import ComfyClient

    address = os.environ.get("COMFYUI_ADDRESS", "127.0.0.1:8188")
    log.info(f"Connecting to ComfyUI at {address}...")
    client = ComfyClient(address)

    # Establish WebSocket — but only require WebUI if explicitly asked
    await client.ensure_connection_async(require_uiapi=require_uiapi)

    _client = client
    log.info(f"ComfyUI client ready (address={address})")
    return _client


# ===========================================================================
# TOOLS
# ===========================================================================


@mcp.tool()
async def check_status() -> dict:
    """Check if ComfyUI is running and a WebUI tab is connected.

    Returns connection status including whether the server is reachable
    and whether a browser tab with the ComfyUI frontend is active.
    This is the first thing to call to verify the setup works.
    """
    try:
        client = await get_client()
        status = await client.json_get_async("/uiapi/connection_status")
        return status
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "hint": "Is ComfyUI running? Start it with: python main.py",
        }


@mcp.tool()
async def query_fields() -> dict:
    """List all editable fields in the currently loaded ComfyUI workflow.

    Returns a structured map of every node and widget that can be read or
    written. Use this to discover field paths before calling get_fields
    or set_fields. Field paths look like "KSampler.seed" or "CLIP Text Encode.text".
    """
    client = await get_client()
    return await client.gets_async()


@mcp.tool()
async def get_fields(paths: list[str]) -> dict:
    """Get current values of specific workflow fields.

    Args:
        paths: List of field paths, e.g. ["KSampler.seed", "CLIP Text Encode.text"].
               Use query_fields first to discover valid paths.
    """
    client = await get_client()
    return await client.get_async(paths)


@mcp.tool()
async def set_fields(fields: dict[str, Any]) -> dict:
    """Set field values in the current workflow.

    Args:
        fields: Mapping of field path to value.
                Example: {"CLIP Text Encode.text": "a cat wearing a top hat",
                          "KSampler.seed": 42}
                Supports strings, numbers, and booleans.
    """
    client = await get_client()
    # Convert dict to list of (path, value) tuples as ComfyClient expects
    field_pairs = list(fields.items())
    result = await client.set_async(field_pairs)
    return result if result is not None else {"status": "ok", "fields_set": len(field_pairs)}


@mcp.tool()
async def get_workflow() -> dict:
    """Get the full workflow JSON in ComfyUI's UI format.

    Returns the complete workflow as currently loaded in the browser,
    including all nodes, connections, and widget values. This is the
    native format used by ComfyUI's frontend.
    """
    client = await get_client()
    result = await client.get_workflow()
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    return result


@mcp.tool()
async def get_workflow_api() -> dict:
    """Get the current workflow in API format (directly submittable to /prompt).

    This is the format needed for execute_workflow. It strips UI metadata
    and returns only the node graph with class_type and inputs.
    """
    client = await get_client()
    result = await client.json_post_async("/uiapi/get_workflow_api")
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    return result


@mcp.tool()
async def connect_nodes(from_path: str, to_path: str) -> dict:
    """Wire two nodes together in the current workflow.

    Args:
        from_path: Output path, e.g. "KSampler.LATENT"
        to_path:   Input path, e.g. "VAE Decode.samples"
    """
    client = await get_client()
    return await client.connect_async(from_path, to_path)


def _numpy_to_mcp_image(result) -> MCPImage:
    """Convert a numpy image array to an MCP-serializable PNG.

    The data from ComfyClient.get_image_async() is already RGB (the BGR→RGB
    conversion happens inside _make_request_once at decode time), so we
    must NOT apply another color-space swap here.
    """
    import numpy as np
    from PIL import Image

    if not isinstance(result, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(result)}")

    if result.dtype in [np.float32, np.float64]:
        result = (result * 255).astype(np.uint8)
    # Data is already RGB from comfy_client — go straight to PIL
    img = Image.fromarray(result)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return MCPImage(data=buf.getvalue(), format="png")


def _extract_best_image(result: dict, workflow: dict) -> MCPImage:
    """Pick the best output image from execute_workflow_async results.

    Uses find_output_node to identify the most downstream SaveImage node,
    then converts its first image to an MCPImage.
    """
    from comfy_client import ComfyClient

    outputs = result.get("outputs", {})
    if not outputs:
        raise ValueError("Execution produced no output images")

    # Try the most downstream output node first
    best_node = ComfyClient.find_output_node(workflow)
    if best_node and best_node in outputs and outputs[best_node]:
        return _numpy_to_mcp_image(outputs[best_node][0])

    # Fallback: first node that has any images
    for node_id, images in outputs.items():
        if images:
            return _numpy_to_mcp_image(images[0])

    raise ValueError("All output nodes returned empty image lists")


@mcp.tool()
async def execute() -> MCPImage:
    """Run the current workflow and return the output image.

    Triggers execution of whatever workflow is loaded in the ComfyUI
    browser tab. Waits for completion and returns the first output image
    from the SaveImage node as a PNG. Typically takes 5-60s depending
    on the workflow complexity and model size.
    """
    import numpy as np

    client = await get_client()
    result = await client.execute_async(wait=True)

    if result is None:
        raise ValueError("Execution produced no output image")

    if isinstance(result, np.ndarray):
        return _numpy_to_mcp_image(result)

    raise ValueError(f"Unexpected result type: {type(result)}")


@mcp.tool()
async def execute_workflow(workflow: dict, fields: Optional[dict[str, Any]] = None) -> dict:
    """Run an arbitrary workflow JSON with optional field overrides.

    Args:
        workflow: Workflow in API format (the format from get_workflow_api).
                  Each key is a node ID mapping to {class_type, inputs}.
        fields:   Optional field overrides. Keys are "NodeTitle.input_name"
                  or just "NodeTitle" (uses first input). Values are the
                  new values to set before execution.

    Returns a dict with prompt_id and base64-encoded output images per node.
    """
    import numpy as np
    from comfy_client import encode_image_to_base64

    client = await get_client()
    result = await client.execute_workflow_async(workflow, fields=fields, wait=True)

    # Convert numpy arrays to base64 PNGs for serialization over MCP
    if isinstance(result, dict) and "outputs" in result:
        serialized_outputs = {}
        for node_id, images in result["outputs"].items():
            encoded = []
            for img in images:
                if isinstance(img, np.ndarray):
                    encoded.append(encode_image_to_base64(img))
                else:
                    encoded.append(str(img))
            serialized_outputs[node_id] = encoded
        result["outputs"] = serialized_outputs

    return result


# ===========================================================================
# Routes — dynamic workflow builders
#
# A route is a strategy for constructing a ComfyUI workflow from common inputs
# (prompt, images). Unlike templates (static frozen graphs), routes build the
# graph shape dynamically — e.g. adding LoadImage nodes per input image,
# batching when there are multiple, wiring into the right generation backend.
#
# Each route is a dict: {name, description, build(client, prompt, images, **params) -> workflow}
# The build function returns an API-format workflow dict ready for execution.
# ===========================================================================

_TEMPLATES_FILE = Path(__file__).parent / "templates.json"


def _load_templates() -> dict[str, dict]:
    """Load saved templates from disk."""
    if _TEMPLATES_FILE.exists():
        try:
            return json.loads(_TEMPLATES_FILE.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Failed to load templates.json: {e}")
    return {}


def _save_templates(templates: dict[str, dict]) -> None:
    """Persist templates to disk."""
    _TEMPLATES_FILE.write_text(json.dumps(templates, indent=2))


# ---------------------------------------------------------------------------
# Image upload helper — routes use this to get server-side filenames
# ---------------------------------------------------------------------------

async def _upload_images(client, image_paths: list[str]) -> list[str]:
    """Upload images to ComfyUI and return server-side filenames."""
    filenames = []
    for i, path in enumerate(image_paths):
        result = await client.upload_image(
            path, folder_type="input", overwrite=True,
            filename=f"generate_input_{i}"
        )
        filenames.append(result["name"])
    return filenames


# ---------------------------------------------------------------------------
# Route: Gemini — multimodal LLM image generation/editing
# ---------------------------------------------------------------------------

async def _build_gemini_workflow(
    client, prompt: str, images: list[str],
    model: Optional[str] = None, **params
) -> dict:
    """Build a Gemini image generation workflow.

    0 images → generate from text prompt
    1 image  → edit/transform with text instruction
    N images → batch inputs, edit/transform with text instruction
    """
    workflow: dict[str, dict] = {}
    nid = 1  # auto-incrementing node ID

    # --- Upload and create LoadImage nodes ---
    image_node_ids: list[str] = []
    if images:
        filenames = await _upload_images(client, images)
        for fname in filenames:
            workflow[str(nid)] = {
                "class_type": "LoadImage",
                "inputs": {"image": fname},
            }
            image_node_ids.append(str(nid))
            nid += 1

    # --- Batch if multiple images ---
    images_input = None
    if len(image_node_ids) > 1:
        batch_inputs = {
            f"images.image{i}": [node_id, 0]
            for i, node_id in enumerate(image_node_ids)
        }
        workflow[str(nid)] = {
            "class_type": "BatchImagesNode",
            "inputs": batch_inputs,
        }
        images_input = [str(nid), 0]
        nid += 1
    elif len(image_node_ids) == 1:
        images_input = [image_node_ids[0], 0]

    # --- GeminiImage2Node ---
    gemini_inputs: dict[str, Any] = {
        "prompt": prompt,
        "model": model or "gemini-3-pro-image-preview",
        "seed": params.get("seed", 0),
        "aspect_ratio": params.get("aspect_ratio", "1:1"),
        "resolution": params.get("resolution", "1K"),
        "response_modalities": "IMAGE",
        "system_prompt": "",
    }
    if images_input is not None:
        gemini_inputs["images"] = images_input

    workflow[str(nid)] = {
        "class_type": "GeminiImage2Node",
        "inputs": gemini_inputs,
        "_meta": {"title": "Gemini Image Generation"},
    }
    gemini_id = str(nid)
    nid += 1

    # --- SaveImage ---
    workflow[str(nid)] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI", "images": [gemini_id, 0]},
    }

    return workflow


# ---------------------------------------------------------------------------
# Route: Diffusion — standard KSampler txt2img pipeline
# ---------------------------------------------------------------------------

async def _build_diffusion_workflow(
    client, prompt: str, images: list[str],
    model: Optional[str] = None, **params
) -> dict:
    """Build a standard diffusion txt2img workflow.

    Checkpoint → CLIP encode → KSampler → VAE decode → SaveImage.
    If no model specified, auto-detects from server.
    """
    if not model:
        model = await _auto_detect_checkpoint(client)

    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]},
              "_meta": {"title": "positive"}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": params.get("negative_prompt", ""), "clip": ["1", 1]},
              "_meta": {"title": "negative"}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {
            "width": params.get("width", 512),
            "height": params.get("height", 512),
            "batch_size": 1,
        }},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": params.get("seed", 0),
            "steps": params.get("steps", 20),
            "cfg": params.get("cfg", 7.0),
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
        }},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["6", 0]}},
    }


async def _auto_detect_checkpoint(client) -> str:
    """Query ComfyUI for available checkpoints; return one or error with the list."""
    try:
        info = await client._make_request_once("GET", "/object_info/CheckpointLoaderSimple")
        node_info = info.get("CheckpointLoaderSimple", {})
        ckpt_list = (node_info
                     .get("input", {})
                     .get("required", {})
                     .get("ckpt_name", [[]])[0])
        if not isinstance(ckpt_list, list) or not ckpt_list:
            raise ValueError(
                "No checkpoint models found on the ComfyUI server. "
                "Download one first with the download_model tool."
            )
        if len(ckpt_list) == 1:
            return ckpt_list[0]
        raise ValueError(
            "Multiple checkpoint models available — specify one with the 'model' parameter:\n"
            + "\n".join(f"  - {m}" for m in ckpt_list)
        )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to query available checkpoints: {e}")


# ---------------------------------------------------------------------------
# Route registry
# ---------------------------------------------------------------------------

_ROUTES: dict[str, dict] = {
    "gemini": {
        "name": "Gemini Image Generation",
        "description": "Generate or edit images with Google Gemini. "
                       "0 images = generate from prompt. "
                       "1+ images = edit/transform with text instruction. "
                       "Params: model (gemini model name), seed, aspect_ratio, resolution.",
        "build": _build_gemini_workflow,
    },
    "diffusion": {
        "name": "Stable Diffusion txt2img",
        "description": "Standard diffusion pipeline: checkpoint → CLIP → KSampler → VAE → save. "
                       "Params: model (checkpoint filename), seed, steps, cfg, width, height, negative_prompt.",
        "build": _build_diffusion_workflow,
    },
}


# ---------------------------------------------------------------------------
# Template & route management tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def save_template(
    template_id: str,
    name: str,
    description: str,
    workflow: Optional[dict] = None,
    image_path: Optional[str] = None,
) -> dict:
    """Save a workflow as a named, reusable template.

    Templates are static workflow snapshots — use them for specific ComfyUI
    workflows you want to reuse exactly. For dynamic workflows that adapt
    to inputs (variable image count, etc.), use routes instead.

    The "default" template ID sets what generate_image uses when no route,
    template, workflow, or image_path is specified.

    Args:
        template_id:  Unique identifier, e.g. "default", "xl_turbo", "inpaint".
        name:         Human-readable name, e.g. "SDXL Turbo Fast Generation".
        description:  What this template does. Shown by list_templates.
        workflow:     API-format workflow JSON.
        image_path:   Path to a ComfyUI PNG — embedded workflow will be extracted.

    Provide exactly one of workflow or image_path.
    """
    from comfy_client import ComfyClient

    if workflow is not None and image_path is not None:
        raise ValueError("Provide either 'workflow' or 'image_path', not both.")
    if workflow is None and image_path is None:
        raise ValueError("Provide either 'workflow' or 'image_path'.")

    if image_path is not None:
        workflow = ComfyClient.extract_workflow_from_png(image_path)

    templates = _load_templates()
    templates[template_id] = {
        "name": name,
        "description": description,
        "workflow": workflow,
    }
    _save_templates(templates)

    return {"status": "ok", "template_id": template_id, "name": name}


@mcp.tool()
async def list_routes() -> dict:
    """List available generation routes and saved templates.

    Routes are dynamic workflow builders that construct the graph at runtime
    based on your inputs (prompt, images, params). Templates are static
    workflow snapshots with overridable fields.

    Use route names with generate_image(route="gemini") and template IDs
    with generate_image(template="my_workflow").
    """
    from comfy_client import ComfyClient

    result: dict[str, Any] = {"routes": {}, "templates": {}}

    for rid, r in _ROUTES.items():
        result["routes"][rid] = {"name": r["name"], "description": r["description"]}

    templates = _load_templates()
    for tid, t in templates.items():
        entry: dict[str, Any] = {"name": t["name"], "description": t["description"]}
        prompts = ComfyClient.extract_prompts(t.get("workflow", {}))
        if prompts:
            entry["prompts"] = prompts
        result["templates"][tid] = entry

    return result


@mcp.tool()
async def delete_template(template_id: str) -> dict:
    """Delete a saved workflow template.

    Args:
        template_id: The template to delete.
    """
    templates = _load_templates()
    if template_id not in templates:
        raise ValueError(f"Template '{template_id}' not found.")
    del templates[template_id]
    _save_templates(templates)
    return {"status": "ok", "deleted": template_id}


# ---------------------------------------------------------------------------
# generate_image — the unified high-level generation tool
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_image(
    prompt: str,
    images: Optional[list[str]] = None,
    route: Optional[str] = None,
    template: Optional[str] = None,
    workflow: Optional[dict] = None,
    image_path: Optional[str] = None,
    model: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fields: Optional[dict[str, Any]] = None,
) -> MCPImage:
    """Generate an image — the single entry point for all ComfyUI generation.

    Fully headless. Three ways to specify what pipeline to run:

    1. **Routes** (dynamic) — build the workflow from scratch based on inputs.
       The graph shape adapts to the number of input images.
       generate_image(prompt="a cat", route="gemini")
       generate_image(prompt="edit this", route="gemini", images=["/path/to/img.png"])
       generate_image(prompt="landscape", route="diffusion", model="sd_xl.safetensors")

    2. **Templates** (static) — use a saved workflow snapshot, override fields.
       generate_image(prompt="a cat", template="my_sdxl_workflow")

    3. **Inline** — pass a workflow dict or a ComfyUI PNG directly.
       generate_image(prompt="a cat", workflow={...})
       generate_image(prompt="redo this", image_path="/outputs/ComfyUI_00042.png")

    Resolution order: route > workflow > image_path > template > "gemini" default.

    Args:
        prompt:           Text prompt / instruction.
        images:           Input image paths (uploaded automatically). Routes handle
                          variable counts — 0 for generation, 1+ for editing.
        route:            Route name: "gemini", "diffusion" (see list_routes).
        template:         Saved template ID (see list_routes).
        workflow:         Inline API-format workflow JSON.
        image_path:       ComfyUI PNG with embedded workflow metadata.
        model:            Model name (checkpoint for diffusion, model ID for Gemini).
        negative_prompt:  Negative prompt (diffusion routes).
        seed:             Generation seed.
        steps:            Sampling steps (diffusion).
        cfg:              CFG scale (diffusion).
        width:            Output width (diffusion).
        height:           Output height (diffusion).
        fields:           Raw workflow field overrides {"node_id.input_name": value}.
                          Highest priority — applied after everything else.

    Returns the output image as PNG.
    """
    import copy
    from comfy_client import ComfyClient

    client = await get_client()

    # Collect route-specific params for the builder
    params = {
        k: v for k, v in {
            "negative_prompt": negative_prompt, "seed": seed, "steps": steps,
            "cfg": cfg, "width": width, "height": height,
        }.items() if v is not None
    }

    # --- Resolve workflow ---
    if route is not None:
        # Dynamic route — build workflow from scratch
        if route not in _ROUTES:
            available = ", ".join(_ROUTES.keys())
            raise ValueError(f"Route '{route}' not found. Available: {available}")
        builder = _ROUTES[route]["build"]
        wf = await builder(client, prompt, images or [], model=model, **params)
    elif workflow is not None:
        wf = workflow
    elif image_path is not None:
        wf = ComfyClient.extract_workflow_from_png(image_path)
    elif template is not None:
        templates = _load_templates()
        if template not in templates:
            available = ", ".join(templates.keys())
            raise ValueError(f"Template '{template}' not found. Available: {available}")
        wf = copy.deepcopy(templates[template]["workflow"])
    else:
        # Default: Gemini route
        builder = _ROUTES["gemini"]["build"]
        wf = await builder(client, prompt, images or [], model=model, **params)

    # --- For template/workflow/image_path modes: apply convenience param overrides ---
    # (Routes already bake params into the workflow during construction)
    if route is None:
        field_map = ComfyClient.analyze_workflow_fields(wf)
        merged_fields: dict[str, Any] = {}

        param_mapping = {
            "positive_prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed, "steps": steps, "cfg": cfg,
            "width": width, "height": height,
        }
        for semantic_name, value in param_mapping.items():
            if value is not None and semantic_name in field_map:
                merged_fields[field_map[semantic_name]] = value

        if fields:
            merged_fields.update(fields)

        if merged_fields:
            # Apply through execute_workflow_async's field handling
            exec_result = await client.execute_workflow_async(
                wf, fields=merged_fields, wait=True
            )
        else:
            exec_result = await client.execute_workflow_async(wf, wait=True)
    else:
        # Route already built everything — just apply raw field overrides if any
        exec_result = await client.execute_workflow_async(
            wf, fields=fields, wait=True
        )

    if not isinstance(exec_result, dict) or "outputs" not in exec_result:
        raise ValueError(f"Unexpected execution result: {exec_result}")

    return _extract_best_image(exec_result, wf)


@mcp.tool()
async def upload_image(path: str, subfolder: Optional[str] = None) -> dict:
    """Upload a local image file to ComfyUI's input directory.

    Args:
        path:      Absolute path to the image file on the MCP client's machine.
        subfolder: Optional subfolder within ComfyUI's input directory.

    Returns the server-side filename that can be used in workflow fields.
    The uploaded image becomes available as a LoadImage node input.
    """
    client = await get_client()
    result = await client.upload_image(path, subfolder=subfolder, folder_type="input", overwrite=True)
    return dict(result)


@mcp.tool()
async def check_model(model_name: str) -> dict:
    """Check if a model file exists locally on the ComfyUI server.

    Args:
        model_name: Name of the model file, e.g. "sd_xl_base_1.0.safetensors"

    Returns status info including whether the model is available and its path.
    """
    client = await get_client()
    return await client.get_model_status_async(model_name)


@mcp.tool()
async def download_model(
    model_name: str,
    url: Optional[str] = None,
    huggingface: Optional[str] = None,
    civitai: Optional[str] = None,
    ckpt_type: str = "checkpoints",
) -> dict:
    """Download a model to the ComfyUI server from URL, HuggingFace, or CivitAI.

    Args:
        model_name: Filename to save as, e.g. "my_model.safetensors"
        url:        Direct download URL
        huggingface: HuggingFace repo path, e.g. "stabilityai/sdxl-turbo/sdxl_turbo.safetensors"
        civitai:    CivitAI URL or AIR tag, e.g. "urn:air:sdxl:checkpoint:civitai:123456@789"
        ckpt_type:  Model type/subfolder: "checkpoints", "loras", "vae", "controlnet", etc.

    Provide exactly one of url, huggingface, or civitai.
    """
    from model_defs import ModelDef

    model_def = ModelDef(
        url=url,
        huggingface=huggingface,
        civitai=civitai,
        ckpt_type=ckpt_type,
    )
    client = await get_client()
    return await client.download_model_async(model_name, model_def)


# ===========================================================================
# RESOURCES
# ===========================================================================

@mcp.resource("comfyui://model-urls")
async def get_model_urls() -> str:
    """Known model download URLs from the local model_urls.json registry.

    This maps model filenames to their download sources (CivitAI, HuggingFace, etc.)
    so you can look up where to get a model that's referenced in a workflow.
    """
    urls_path = Path(__file__).parent / "model_urls.json"
    if urls_path.exists():
        return urls_path.read_text()
    return json.dumps({})


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    # Force all logging to stderr before entering the MCP event loop.
    # FastMCP's configure_logging() uses basicConfig() which may attach
    # a handler to stdout — we override after the fact.
    root = logging.getLogger()
    root.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    root.addHandler(h)
    root.setLevel(logging.INFO)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
