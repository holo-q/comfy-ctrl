import asyncio
from calendar import c
import logging
import time
import json
import traceback
from typing import Dict, Any, Optional, Callable, Awaitable, Tuple, List
import base64
from pathlib import Path
import os
import uuid
from dataclasses import dataclass
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install

# ANSI color codes
# Regular colors
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

# Bright colors
BRIGHT_BLACK = "\033[90m"
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"

# Background colors
BG_BLACK = "\033[40m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"
BG_WHITE = "\033[47m"

# Bright background colors
BG_BRIGHT_BLACK = "\033[100m"
BG_BRIGHT_RED = "\033[101m"
BG_BRIGHT_GREEN = "\033[102m"
BG_BRIGHT_YELLOW = "\033[103m"
BG_BRIGHT_BLUE = "\033[104m"
BG_BRIGHT_MAGENTA = "\033[105m"
BG_BRIGHT_CYAN = "\033[106m"
BG_BRIGHT_WHITE = "\033[107m"

# Text style codes
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
ITALIC = "\033[3m"
STRIKE = "\033[9m"
RESET = "\033[0m"

LOG_WEBUI_RESPOND = f"{BLUE}(webui:respond){RESET}"
LOG_WEBUI_SEND = f"{BLUE}(webui:send){RESET}"
LOG_WEBUI_CONNECT = f"{BLUE}(webui){RESET}"
LOG_WEBUI_DISCONNECT = f"{BLUE}(webui){RESET}"
LOG_TASK_START = f"{MAGENTA}(task:start){RESET}"
LOG_TASK_END = f"{MAGENTA}(task:end){RESET}"
LOG_TASK = f"{MAGENTA}(task){RESET}"
LOG_MODEL = f"{GREEN}(model){RESET}"

from aiohttp import web
from aiohttp import web_response
from aiohttp.web import RouteTableDef

import server
from .model_defs import ModelDef
from . import model_defs
from server import PromptServer

# Path to store model URLs
MODEL_URLS_PATH = Path(__file__).parent / "model_urls.json"
INF_TIMEOUT = 99999999999

def load_stored_urlmap() -> Dict[str, str]:
    """Load saved model URLs from JSON file"""
    if MODEL_URLS_PATH.exists():
        try:
            with open(MODEL_URLS_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Error loading model URLs: {e}")
    return {}


def save_model_urls(urls: Dict[str, str]) -> None:
    """Save model URLs to JSON file, merging with existing URLs"""
    try:
        existing_urls = load_stored_urlmap()
        merged_urls = {**existing_urls, **urls}

        os.makedirs(MODEL_URLS_PATH.parent, exist_ok=True)
        with open(MODEL_URLS_PATH, "w") as f:
            json.dump(merged_urls, f, indent=2)

        log.info(f"Saved {len(merged_urls)} model URLs ({len(urls)} new/updated)")
    except Exception as e:
        log.error(f"Error saving model URLs: {e}")


# Mock routes if server not initialized
routes = PromptServer.instance.routes if hasattr(PromptServer, 'instance') and PromptServer.instance else RouteTableDef()
server_start_time = time.time()

# Install rich traceback handler
install(show_locals=True)

# Set up rich console
console = Console()

# Configure logging with rich
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True, markup=True, show_time=True, show_path=True)
    ],
)

log = logging.getLogger("uiapi")

# Remove any existing handlers to avoid duplicate logging
for handler in log.handlers[:]:
    log.removeHandler(handler)

# Add rich handler with markup enabled
# rich_handler = RichHandler(
#     console=console,
#     rich_tracebacks=True,
#     markup=True,
#     show_time=True,
#     show_path=True,
#     enable_link_path=False,
# )
# log.addHandler(rich_handler)
# log.propagate = False


# Set up handlers to pass over to the client and then back.
# This python server extension is simply a middleman from the
# other client which connects to the server and our own webui client
# ------------------------------------------

INPUT_DIR = Path(__file__).parent.parent.parent / "input"

NEXT_CLIENT_ID = 0


@dataclass
class PendingRequest:
    """Represents a request that's waiting for client response"""

    request_id: str
    endpoint: str
    data: dict
    event: asyncio.Event
    response: Any = None
    post_process: Optional[Callable[[Any], Awaitable[Any]]] = None


class WebuiManager:
    # Class-level dictionary to store all client managers
    _client_managers: Dict[int, "WebuiManager"] = {}
    _main_webui: Optional["WebuiManager"] = None

    def __init__(self):
        self._pending_requests: Dict[str, PendingRequest] = {}
        self._client_id: Optional[int] = None
        self._webui_ready = asyncio.Event()
        self._buffered_requests: list[PendingRequest] = []
        self._lock = asyncio.Lock()
        log.info(f"{LOG_WEBUI_CONNECT} - *")

    @property
    def client_id(self) -> Optional[int]:
        """Get the client ID for this manager"""
        return self._client_id

    @classmethod
    def get_or_create_manager(cls, client_id: int) -> "WebuiManager":
        """Get an existing manager or create a new one for a client ID"""
        id = int(client_id)

        if id == -1:
            global NEXT_CLIENT_ID
            NEXT_CLIENT_ID += 1

            # Create new manager and generate UUID
            manager = cls()
            manager._client_id = NEXT_CLIENT_ID
            cls._client_managers[NEXT_CLIENT_ID] = manager
            return manager

        if id in cls._client_managers:
            return cls._client_managers[id]

        # Create new manager with provided ID
        manager = cls()
        manager._client_id = id
        cls._client_managers[id] = manager
        return manager

    @classmethod
    def get_main_webui(cls) -> Optional["WebuiManager"]:
        """Get the main WebUI manager instance"""
        return cls._main_webui

    def set_connected(self):
        """Called when ComfyUI web interface connects"""
        self._webui_ready.set()

        # Set this as the main WebUI if none exists or current main is disconnected
        if (
            not self.__class__._main_webui
            or not self.__class__._main_webui._webui_ready.is_set()
        ):
            log.info(f"{LOG_WEBUI_CONNECT} [{self._client_id}] - Setting as main WebUI")
            self.__class__._main_webui = self

        # Process buffered requests
        if len(self._buffered_requests) > 0:
            log.info(
                f"{LOG_WEBUI_CONNECT} [{self._client_id}] - {len(self._buffered_requests)} requests in buffer"
            )
            for req in self._buffered_requests:
                log.info(f"Processing buffered request: {req.endpoint}")
        else:
            log.info(f"{LOG_WEBUI_CONNECT} [{self._client_id}]")

        self._buffered_requests.clear()

    def set_disconnected(self):
        """Called when ComfyUI web interface disconnects"""
        log.warning(f"{LOG_WEBUI_DISCONNECT} [{self._client_id}]")
        self._webui_ready.clear()

        # If this was the main WebUI, try to find another connected one
        if self.__class__._main_webui == self:
            self.__class__._main_webui = None
            for manager in self.__class__._client_managers.values():
                if manager != self and manager._webui_ready.is_set():
                    log.info(
                        f"{LOG_WEBUI_CONNECT} [{manager._client_id}] - Setting as new main WebUI"
                    )
                    self.__class__._main_webui = manager
                    break

        if self._client_id in self._client_managers:
            del self._client_managers[self._client_id]

    async def _send(self, request: PendingRequest):
        """Send a request to the client"""
        data_str = json.dumps(request.data, indent=2).replace("\n", " ")
        log.info(
            f"{LOG_WEBUI_SEND} [{self._client_id}] {request.request_id} {request.endpoint} {data_str[:100]}{'...' if len(data_str) > 100 else ''}"
        )
        # Add client_id to request data
        request.data["client_id"] = self._client_id
        await server.PromptServer.instance.send_json(request.endpoint, request.data)

    async def send(
        self,
        endpoint: str,
        data: dict | None = None,
        buffered: bool = False,
        post_process: Optional[Callable[[Any], Awaitable[Any]]] = None,
    ) -> PendingRequest:
        """Create a new request and optionally buffer it"""
        request_id = str(uuid.uuid4())[:8]
        endpoint = f"/uiapi/{endpoint}"

        if data is None:
            data = {}
        data["request_id"] = request_id
        data["client_id"] = self._client_id

        request = PendingRequest(
            request_id=request_id,
            endpoint=endpoint,
            data=data,
            event=asyncio.Event(),
            post_process=post_process,
        )

        async with self._lock:
            self._pending_requests[request_id] = request
            if buffered and not self._webui_ready.is_set():
                log.info(
                    f"{LOG_WEBUI_SEND} [{self._client_id}] {request_id} - WebUI not ready, buffering ..."
                )
                self._buffered_requests.append(request)
                return request

        await self._send(request)
        return request

    async def wsend(
        self,
        endpoint: str,
        data: dict | None = None,
        buffered: bool = False,
        post_process: Optional[Callable[[Any], Awaitable[Any]]] = None,
    ) -> Any:
        """Send a request to the client and wait for a response"""
        request = await self.send(endpoint, data, buffered, post_process)
        return await self.wait(request)

    async def respond(self, request_id: str, response_data: Any):
        """Set response for a request and process it"""
        response_str = json.dumps(response_data, indent=2)
        log.info(
            f"{LOG_WEBUI_RESPOND} [{self._client_id}] {request_id} {response_str[:100]}{'...' if len(response_str) > 100 else ''}"
        )

        async with self._lock:
            if (
                request_id not in self._pending_requests
                or self._pending_requests[request_id].response
            ):
                log.warning(
                    f"{LOG_WEBUI_RESPOND} [{self._client_id}] response received for unknown request: {request_id}"
                )
                return

            request = self._pending_requests[request_id]
            request.response = response_data

            if request.post_process:
                log.info(f"Running post-processing for request {request_id}")
                try:
                    request.response = await request.post_process(response_data)
                    log.info(
                        f"Post-processed response: {json.dumps(request.response, indent=2)}"
                    )
                except Exception as e:
                    log.error(f"Error in post-processing for {request_id}: {e}")
                    log.error(traceback.format_exc())

            request.event.set()
            log.debug(f"{LOG_WEBUI_RESPOND} [{self._client_id}] {request_id} complete")

    async def wait(self, request: PendingRequest, timeout: float = 30.0) -> Any:
        """Wait for and return response for a request"""
        log.debug(
            f"{LOG_WEBUI_SEND} [{self._client_id}] {request.request_id} (timeout={timeout}s)"
        )
        try:
            await asyncio.wait_for(request.event.wait(), timeout)
            return request.response
        except asyncio.TimeoutError:
            log.error(
                f"{LOG_WEBUI_SEND} [{self._client_id}] {request.request_id} timed out after {timeout}s"
            )
        finally:
            async with self._lock:
                self._pending_requests.pop(request.request_id, None)
                log.debug(
                    f"{LOG_WEBUI_SEND} [{self._client_id}] {request.request_id} popped"
                )


# Initialize the request manager
webui_manager = WebuiManager()


async def handle_uiapi_request(
    endpoint: str,
    request_data: dict | None = None,
    wait_for_client: bool = False,
    post_process: Optional[Callable[[Any], Awaitable[Any]]] = None,
) -> web.Response:
    """Handle a UI API request with optional post-processing"""
    log.info(f"-> /uiapi/{endpoint}")
    log.info(
        f"-> /uiapi/{endpoint} {json.dumps(request_data, indent=2) if request_data else 'None'}"
    )

    try:
        # Get the main WebUI manager
        main_webui = WebuiManager.get_main_webui()
        if not main_webui:
            return web.json_response(
                {"status": "error", "error": "No main WebUI connected"}, status=503
            )

        request = await main_webui.send(
            endpoint, request_data, wait_for_client, post_process
        )

        if request in main_webui._buffered_requests:
            log.info(f"{LOG_WEBUI_SEND} {request.request_id} - buffered")
            return web.json_response(
                {
                    "status": "pending",
                    "message": "Request buffered until client connects",
                    "request_id": request.request_id,
                }
            )

        response = await main_webui.wait(request, timeout=30.0)
        log.info(f"Request completed successfully {request.request_id}")
        return web.json_response({"status": "ok", "response": response})
    except asyncio.TimeoutError:
        log.error(f"{LOG_WEBUI_SEND} {request.request_id} - timeout")
        return web.json_response(
            {"status": "error", "error": "Request timed out"}, status=408
        )
    except Exception as e:
        log.error(f"{LOG_WEBUI_SEND} {request.request_id} - {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


# Track ongoing downloads
download_tasks: Dict[str, Dict[str, Any]] = {}


def analyze_workflow_models(workflow: dict) -> Tuple[List[str], List[str], List[str]]:
    """Analyze a workflow and return lists of missing, existing, and all checkpoints.

    Args:
        workflow: The workflow to analyze

    Returns:
        Tuple of (missing_models, existing_models, all_checkpoints)
    """
    checkpoints = []
    missing_models = []
    existing_models = []

    # Extract nodes from workflow
    nodes = workflow.get("nodes", {})

    for node in nodes:
        if isinstance(node, dict) and "widgets_values" in node:
            for value in node["widgets_values"]:
                if isinstance(value, str) and any(
                    value.endswith(ext) for ext in [".safetensors", ".ckpt", ".bin"]
                ):
                    checkpoints.append(value)
                    name = value.split("/")[-1]
                    if model_defs.has_model(name):
                        existing_models.append(name)
                    else:
                        missing_models.append(value)

    return missing_models, existing_models, checkpoints


@routes.post("/uiapi/webui_ready")
async def uiapi_webui_ready(request):
    """Called when ComfyUI web interface connects"""
    print()
    data = await request.json()
    client_id = data.get("client_id", "-1")
    browser_info = data.get("browserInfo", {})
    browser_name = browser_info.get("browser", "Unknown Browser")
    platform = browser_info.get("platform", "Unknown Platform")

    # Get or create manager for this client
    print("")
    log.info("-> /uiapi/webui_ready")

    manager = WebuiManager.get_or_create_manager(client_id)
    log.info(
        f"{LOG_WEBUI_CONNECT} [{manager.client_id}] - Hello from {browser_name} on {platform}!"
    )
    manager.set_connected()

    try:
        # Fetch current workflow using this manager since it just connected
        workflow = await manager.wsend("get_workflow", buffered=True)

        if workflow:
            workflow = workflow["workflow"]["workflow"]
            missing_models, existing_models, checkpoints = analyze_workflow_models(
                workflow
            )

            console.print()
            console.print(
                f"{BOLD}{BLUE}========== Workflow Analysis Results =========={RESET}"
            )
            console.print(f"{LOG_MODEL} Total checkpoints found: {len(checkpoints)}")

            console.print(f"{LOG_MODEL} Missing models: {len(missing_models)}")
            if missing_models:
                for model in missing_models:
                    console.print(f"{LOG_MODEL} • {model}")

            console.print(f"{LOG_MODEL} Existing models: {len(existing_models)}")
            if existing_models:
                for model in existing_models:
                    console.print(f"{LOG_MODEL} • {model}")
        else:
            console.print(f"{YELLOW}No workflow available to analyze{RESET}")

    except Exception as e:
        log.error(f"Error analyzing workflow: {e}")
        log.error(traceback.format_exc())

    return web.json_response({"status": "ok", "client_id": manager.client_id})


@routes.post("/uiapi/webui_disconnect")
async def uiapi_webui_disconnect(request):
    """Called when ComfyUI web interface disconnects"""
    data = await request.json()
    client_id = data.get("client_id")
    if client_id:
        manager = WebuiManager.get_or_create_manager(client_id)
        manager.set_disconnected()
    return web.json_response({"status": "ok"})


@routes.post("/uiapi/webui_response")
async def uiapi_response(request):
    data = await request.json()
    request_id = data.get("request_id")
    client_id = data.get("client_id")

    if not request_id:
        return web.json_response(
            {"status": "error", "error": "No request_id provided"}, status=400
        )

    if not client_id:
        return web.json_response(
            {"status": "error", "error": "No client_id provided"}, status=400
        )

    manager = WebuiManager.get_or_create_manager(client_id)
    await manager.respond(request_id, data.get("response"))
    return web.json_response({"status": "ok"})


# Example of download_models with post-processing
async def process_workflow_response(response: Any) -> Any:
    """Process workflow response for download_models"""
    if isinstance(response, dict):
        # Handle pending status
        if response.get("status") == "pending":
            return response
        # Handle normal response
        return response.get("response", {})
    elif isinstance(response, web.Response):
        # Get response text directly from the response object
        try:
            if hasattr(response, "text"):
                if callable(response.text):
                    response_text = await response.text()
                else:
                    response_text = response.text

                if isinstance(response_text, str):
                    response_data = json.loads(response_text)
                    if "workflow" in response_data:
                        return response_data["workflow"]
        except Exception as e:
            log.error(f"Error processing workflow response: {e}")
            log.error(traceback.format_exc())
    return response


# Add this near other global variables
pending_downloads: Dict[str, Dict] = {}


@routes.post("/uiapi/download_models")
async def uiapi_download_models(request):
    try:
        print()
        log.info("-> /uiapi/download_models")
        # Get the request data containing the download table
        request_data = await request.json()
        download_table = request_data.get("download_table", {})
        workflow = request_data.get("workflow", None)
        
        # Get workflow from main WebUI if not provided
        # ----------------------------------------
        if not workflow:
            log.info("/uiapi/download_models - No workflow provided, checking main WebUI")
            main_webui = WebuiManager.get_main_webui()
            if main_webui:
                workflow = await main_webui.wsend("get_workflow", buffered=True)
                if workflow:
                    workflow = workflow["workflow"]["workflow"]
                    log.info("/uiapi/download_models - Got workflow from main WebUI")
                else:
                    log.info("/uiapi/download_models - Failed to get workflow from main WebUI")
                    return web.json_response(
                        {"status": "error", "error": "No workflow available"}, status=400
                    )
            else:
                log.info("/uiapi/download_models - No main WebUI available")
                return web.json_response(
                    {"status": "error", "error": "No workflow provided and no main WebUI available"}, status=400
                )

        # Create task
        # ----------------------------------------

        task_id = str(uuid.uuid4())[:8]
        download_tasks[task_id] = {
            "status": "initializing",
            "start_time": time.time(),
            "progress": {},
            "completed": False,
        }
        download_task = download_models_task(task_id, download_table, workflow)
        asyncio.create_task(download_task)

        log.info(f"{LOG_TASK_START} {task_id} - *")

        return web.json_response(
            {"status": "ok", "message": "Download task started", "download_id": task_id}
        )

    except Exception as e:
        log.error(f"{LOG_TASK_END} Download task error: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.get("/uiapi/download_status/{request_id}")
async def uiapi_download_status(request):
    """Handle status requests for downloads in progress"""
    request_id = request.match_info["request_id"]
    log.info(f"-> /uiapi/download_status/{request_id}")

    # First check pending downloads
    if request_id in pending_downloads:
        # Clean up old pending downloads (older than 5 minutes)
        current_time = time.time()
        for rid, info in list(pending_downloads.items()):
            if current_time - info["timestamp"] > 300:  # 5 minutes
                pending_downloads.pop(rid)

        # Return status if still pending
        if request_id in pending_downloads:
            info = pending_downloads[request_id]
            # If task_id is set, redirect to that status
            if info["task_id"]:
                return web.json_response(download_tasks[info["task_id"]])
            return web.json_response(info)

    # Then check active download tasks
    if request_id in download_tasks:
        return web.json_response(download_tasks[request_id])

    return web.json_response({"status": "error", "error": "Task not found"}, status=404)


@routes.post("/uiapi/get_workflow")
async def uiapi_get_workflow(request):
    print()
    log.info("-> /uiapi/get_workflow")
    return await handle_uiapi_request("get_workflow")


@routes.post("/uiapi/get_workflow_api")
async def uiapi_get_workflow_api(request):
    print()
    log.info("-> /uiapi/get_workflow_api")
    return await handle_uiapi_request("get_workflow_api")


@routes.post("/uiapi/get_fields")
async def uiapi_get_field(request):
    print()
    log.info("-> /uiapi/get_fields")
    return await handle_uiapi_request("get_fields", await request.json())


def save_base64_image(base64_data: str, filepath: Path) -> None:
    """Save base64 image data to a file"""
    try:
        image_data = base64.b64decode(base64_data)
        os.makedirs(filepath.parent, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(image_data)
    except Exception as e:
        log.error(f"Failed to save base64 image: {e}")
        import traceback

        log.error(traceback.format_exc())
        raise


@routes.post("/uiapi/set_fields")
async def uiapi_set_field(request):
    request_data = await request.json()
    print()
    log.info("-> /uiapi/set_fields")

    # Process any image fields before passing to the main handler
    if "fields" in request_data:
        for field in request_data["fields"]:
            if isinstance(field[1], dict) and field[1].get("type") == "image_base64":
                path = field[0]
                # Generate input name from the node path
                input_name = f'INPUT_{path.split(".")[0]}.png'

                # TODO: Verify this is the correct path for ComfyUI's input folder
                filepath = INPUT_DIR / input_name

                # Save the base64 image
                save_base64_image(field[1]["data"], filepath)

                # Replace the base64 data with just the filename
                field[1] = input_name

    return await handle_uiapi_request("set_fields", request_data)


@routes.post("/uiapi/set_connection")
async def uiapi_set_connection(request):
    print()
    log.info("-> /uiapi/set_connection")
    return await handle_uiapi_request("set_connection", await request.json())


@routes.post("/uiapi/execute")
async def uiapi_execute(request):
    print()
    log.info("-> /uiapi/execute")
    return await handle_uiapi_request("execute", await request.json())


@routes.post("/uiapi/query_fields")
async def uiapi_query_fields(request):
    print()
    log.info("-> /uiapi/query_fields")
    return await handle_uiapi_request("query_fields", await request.json())


@routes.post("/uiapi/add_model_url")
async def uiapi_add_model_url(request):
    """Add a URL for a model that wasn't in the download table"""
    print()
    log.info("-> /uiapi/add_model_url")
    try:
        data = await request.json()
        model_name = data.get("model_name")
        model_url = data.get("url")
        model_type = data.get("model_type", "model")  # Default to 'model' type

        if not model_name or not model_url:
            return web.json_response(
                {"status": "error", "error": "Missing model_name or url"}, status=400
            )

        # Create model definition
        model_def = {"url": model_url, "ckpt_type": model_type}

        return web.json_response({"status": "ok", "model_def": model_def})

    except Exception as e:
        log.error(f"Error adding model URL: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


async def download_models_task(
    task_id: str, urlmap_request: Dict[str, dict], workflow: dict
) -> None:
    """Background task to handle model downloads"""
    log.info(f"{LOG_TASK_START} {task_id} - Starting")

    task_info = download_tasks[task_id]
    start_time = time.time()

    try:
        # Determine missing models
        # ----------------------------------------

        checkpoints = []
        missing_ckpts = []
        existing_models = []
        ckpt_types = {}

        task_info.update(
            {
                "status": "checking",
                "progress": {},
            }
        )

        # Read workflow nodes
        # ----------------------------------------
        def process_model_value(value, ntype, task_id):
            """Process a single model value and update tracking data"""
            if not (isinstance(value, str) and any(
                value.endswith(ext) for ext in [".safetensors", ".ckpt", ".bin"]
            )):
                return None
                
            ckpt = value
            name = ckpt.split("/")[-1]
            
            # Determine checkpoint type
            if 'lora' in ntype or 'loraloader' in ntype:
                ckpt_types[ckpt] = 'loras'
            elif 'control' in ntype:
                ckpt_types[ckpt] = 'controlnet'
            else:
                ckpt_types[ckpt] = 'checkpoints'

            checkpoints.append(ckpt)
            
            # Check if model exists
            if model_defs.has_model(ckpt, ckpt_types[ckpt]):
                path = model_defs.get_model_path(ckpt, ckpt_types[ckpt])
                log.info(
                    f"{LOG_TASK} {task_id} - {GREEN} ✓ {ckpt_types[ckpt]}: {ckpt}{RESET}"
                )
                existing_models.append(ckpt)
                task_info["progress"][value] = {
                    "status": "success",
                    "path": str(path),
                    "model_number": "existing",
                }
            else:
                missing_ckpts.append(ckpt)
                log.info(
                    f"{LOG_TASK} {task_id} - {RED} ✗ {ckpt_types[ckpt]}: {ckpt}{RESET}"
                )

        # 1) UI json format
        nodes = workflow.get("nodes", [])
        if nodes:
            log.info(
                f"{LOG_TASK_START} {task_id} - Extracted {len(nodes)} nodes from workflow"
            )
        for node in nodes:
            if isinstance(node, dict) and "widgets_values" in node:
                for value in node["widgets_values"]:
                    process_model_value(value, node["type"].lower(), task_id)

        # 2) API json format
        if not nodes and isinstance(workflow, dict):
            log.info(f"{LOG_TASK_START} {task_id} - Processing alternate workflow format")
            for node_id, node_data in workflow.items():
                if isinstance(node_data, dict):
                    # Check inputs for model paths
                    inputs = node_data.get('inputs', {})
                    for key, value in inputs.items():
                        process_model_value(value, node_data.get('class_type', '').lower(), task_id)

        # Tally models
        # ----------------------------------------
        total_models = len(missing_ckpts)
        if total_models == 0:
            log.info(
                f"{LOG_TASK_END} {task_id} - All models already exist, nothing to download"
            )
            task_info.update(
                {
                    "status": "completed",
                    "completed": True,
                    "elapsed_time": time.time() - start_time,
                }
            )
            return

        # Download models
        # ----------------------------------------
        log.info(
            f"{LOG_TASK_START} {task_id} - Found {total_models} models to download"
        )

        task_info.update(
            {
                "total_models": total_models,
                "current_model": 0,
                "status": "downloading",
            }
        )

        # Load saved URLs
        urlmap_store = load_stored_urlmap()
        log.info(
            f"{LOG_TASK_START} {task_id} - Loaded {len(urlmap_store)} saved model URLs"
        )

        # Get URLs from client not in the download map
        urlmap_both = {**urlmap_store, **urlmap_request}
        nourl_ckpts = [ckpt for ckpt in missing_ckpts if ckpt not in urlmap_both]
        hasurl_ckpts = [ckpt for ckpt in missing_ckpts if ckpt in urlmap_both]
        webui = WebuiManager.get_main_webui()
        if len(nourl_ckpts) > 0 and webui is not None:
            log.info(
                f"{LOG_TASK_START} {task_id} - Requesting urlmap from webui for {len(nourl_ckpts)} models ..."
            )
            request = await webui.wsend(
                "get_model_url",
                {
                    "requested_ckpts": nourl_ckpts,
                    "existing_ckpts": hasurl_ckpts,
                },
                buffered=True,
            )

            # Wait for response from webui
            urlmap_webui = await webui.wait(request, timeout=INF_TIMEOUT)
            if not urlmap_webui:
                urlmap_webui = {}
                log.warning(
                    f"{LOG_TASK_END} {task_id} - No urlmap provided, the following models will be missing and cause the workflow not to run:"
                )
                for model in missing_ckpts:
                    log.warning(f"    {model}")
            else:
                # Combine saved URLs with new ones
                for name in missing_ckpts:
                    if name in urlmap_webui and urlmap_webui[name]:
                        urlmap_store[name] = urlmap_webui[name]

                # Save updated URLs
                save_model_urls(urlmap_store)
                log.info(
                    f"{LOG_TASK_END} {task_id} - Saved {len(urlmap_store)} new model URLs"
                )

                # Use combined URL map for downloads
                urlmap = {**urlmap_store, **urlmap_request, **urlmap_webui}
        else:
            urlmap = {**urlmap_store, **urlmap_request}

        # Process downloads one at a time
        for idx, ckpt in enumerate(missing_ckpts, 1):
            name = ckpt.split("/")[-1]
            elapsed = time.time() - start_time

            # log.info(
            #     f"{LOG_TASK_START} {task_id} - Processing model {idx}/{len(missing_models)}: {ckpt}"
            # )

            task_info["current_model"] = idx
            task_info["elapsed_time"] = elapsed
            task_info["progress"][ckpt] = {
                "status": "downloading",
                "model_number": f"{idx}/{len(missing_ckpts)}",
            }

            try:
                url = urlmap.get(ckpt) or urlmap.get(name)
                if isinstance(url, dict):
                    model_def = ModelDef(**url)
                elif isinstance(url, str):
                    model_def = ModelDef(url=url)
                else:
                    log.warning(
                        f"{LOG_TASK_END} {task_id} - No URL provided for {ckpt}, skipping..."
                    )
                    task_info["progress"][ckpt].update(
                        {"status": "error", "error": "No download URL provided"}
                    )
                    continue

                path = model_def.download(ckpt, type=ckpt_types[ckpt])
                task_info["progress"][ckpt].update(
                    {"status": "success", "path": str(path)}
                )

            except Exception as e:
                log.error(f"{LOG_TASK_END} {task_id} - Error downloading {ckpt}: {e}")
                log.error(traceback.format_exc())
                task_info["progress"][ckpt].update({"status": "error", "error": str(e)})

        task_info["status"] = "completed"
        task_info["completed"] = True
        task_info["elapsed_time"] = time.time() - start_time
        log.info(
            f"{LOG_TASK_END} {task_id} - Completed in {task_info['elapsed_time']:.1f}s"
        )

    except Exception as e:
        log.error(f"{LOG_TASK_END} {task_id} - Failed: {e}")
        log.error(traceback.format_exc())
        task_info["status"] = "error"
        task_info["error"] = str(e)


@routes.get("/uiapi/connection_status")
async def uiapi_connection_status(request):
    """Check if ComfyUI web interface is connected and get system status"""
    try:
        client_id = request.query.get("client_id")
        if client_id:
            manager = WebuiManager.get_or_create_manager(client_id)
            status = {
                "status": "ok",
                "webui_connected": manager._webui_ready.is_set(),
                "pending_requests": len(manager._buffered_requests),
                "client_id": manager.client_id,
                "is_main": manager == WebuiManager.get_main_webui(),
                "server_time": time.time(),
                "server_uptime": time.time() - server_start_time,
            }
        else:
            # Return overall system status
            main_webui = WebuiManager.get_main_webui()
            status = {
                "status": "ok",
                "active_clients": len(WebuiManager._client_managers),
                "main_webui_id": main_webui.client_id if main_webui else None,
                "active_downloads": len(download_tasks),
                "pending_downloads": len(pending_downloads),
                "server_time": time.time(),
                "server_uptime": time.time() - server_start_time,
            }

        log.debug("-> /uiapi/connection_status")
        return web.json_response(status)
    except Exception as e:
        log.error(f"Error in connection status check: {e}")
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.post("/uiapi/client_disconnect")
async def uiapi_client_disconnect(request):
    """Handle explicit client disconnect notification"""
    try:
        data = await request.json()
        client_id = data.get("client_id")
        if client_id:
            manager = WebuiManager.get_or_create_manager(client_id)
            manager.set_disconnected()
        return web.json_response({"status": "ok"})
    except Exception as e:
        log.error(f"Error handling client disconnect: {e}")
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.post("/uiapi/download_model")
async def uiapi_download_model(request):
    """Handle single model download request"""
    try:
        data = await request.json()
        model_name = data.get("model_name")
        model_def = data.get("model_def")

        if not model_name or not model_def:
            return web.json_response(
                {"status": "error", "error": "Missing model_name or model_def"},
                status=400,
            )

        # Create ModelDef instance
        try:
            model = ModelDef(**model_def)
        except Exception as e:
            return web.json_response(
                {"status": "error", "error": f"Invalid model definition: {str(e)}"},
                status=400,
            )

        # Start download
        try:
            path = model.download(model_name)
            return web.json_response({"status": "success", "path": str(path)})
        except Exception as e:
            return web.json_response({"status": "error", "error": str(e)}, status=500)

    except Exception as e:
        log.error(f"<error> download model error: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.get("/uiapi/model_status/{model_name}")
async def uiapi_model_status(request):
    """Get status of a specific model"""
    model_name = request.match_info["model_name"]

    try:
        # Check if model exists
        if model_defs.has_model(model_name):
            path = model_defs.get_model_path(model_name)
            return web.json_response(
                {
                    "status": "success",
                    "exists": True,
                    "path": str(path),
                }
            )

        # Check if download is in progress
        if model_name in download_tasks:
            task = download_tasks[model_name]
            return web.json_response(
                {
                    "status": "downloading",
                    "progress": task.get("progress", 0),
                    "started": task.get("start_time"),
                    "elapsed": time.time() - task.get("start_time", time.time()),
                }
            )

        return web.json_response({"status": "not_found", "exists": False})

    except Exception as e:
        log.error(f"<error> model status error: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.post("/uiapi/get_model_url")
async def uiapi_get_model_url(request):
    """Handle requests to get a model's download URL"""
    webui = WebuiManager.get_main_webui()
    assert webui is not None

    try:
        data = await request.json()
        ckpt_name = data.get("ckpt_name")
        ckpt_names = data.get("ckpt_names")
        print("")
        log.info("-> /uiapi/get_model_url")
        log.info(f"-> /uiapi/get_model_url: {data}")

        # Handle batch request
        if ckpt_names:
            if not isinstance(ckpt_names, list):
                return web.json_response(
                    {"status": "error", "error": "ckpt_names must be a list"},
                    status=400,
                )
            request = await webui.wsend(
                "get_model_url", {"ckpt_names": ckpt_names}, buffered=True
            )
            response = await webui.wait(request, timeout=INF_TIMEOUT)

            # Save URLs to persistent storage
            if response:
                saved_urls = load_stored_urlmap()
                for name, url in response.items():
                    if url:  # Only save non-empty URLs
                        saved_urls[name] = url
                save_model_urls(saved_urls)
                log.info(f"Saved {len(response)} model URLs")

            return web.json_response({"status": "ok", "urls": response})
        elif ckpt_name:
            # Create a request to get URL from webui
            request = await webui.wsend(
                "get_model_url", {"ckpt_name": ckpt_name}, buffered=True
            )

            # Wait for response from webui
            response = await webui.wait(request, timeout=INF_TIMEOUT)

            if not response or not response.get("url"):
                return web.json_response(
                    {"status": "error", "error": "No URL provided"}, status=400
                )

            # Save URL to persistent storage
            saved_urls = load_stored_urlmap()
            saved_urls[ckpt_name] = response["url"]
            save_model_urls(saved_urls)
            log.info(f"-> /uiapi/get_model_url: saved URL for model {ckpt_name}")

            return web.json_response(
                {
                    "status": "ok",
                    "url": response["url"],
                    "model_type": response.get("model_type", "model"),
                }
            )
        else:
            return web.json_response(
                {"status": "error", "error": "Missing ckpt_name"}, status=400
            )

    except Exception as e:
        log.error(f"Error getting model URL: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.post("/uiapi/send_workflow")
async def uiapi_send_workflow(request):
    """Send a workflow to all connected webuis for user approval"""
    print("")
    log.info("-> /uiapi/send_workflow")
    try:
        data = await request.json()
        workflow = data.get("workflow")
        message = data.get("message", "Would you like to load this workflow?")
        title = data.get("title", "Load Workflow")

        if not workflow:
            return web.json_response(
                {"status": "error", "error": "No workflow provided"}, status=400
            )

        # Get all connected WebUI managers
        connected_managers = [
            m for m in WebuiManager._client_managers.values() if m._webui_ready.is_set()
        ]

        if not connected_managers:
            return web.json_response(
                {"status": "error", "error": "No WebUI clients connected"}, status=400
            )

        log.info(
            f"Sending workflow dialog to {len(connected_managers)} connected WebUIs"
        )

        # Send to all connected WebUIs and wait for first accept
        responses = []
        for manager in connected_managers:
            try:
                # Create a request to show dialog in webui
                request = await manager.send(
                    "show_workflow_dialog",
                    {"workflow": workflow, "message": message, "title": title},
                    buffered=True,
                )

                # Wait for user response
                response = await manager.wait(request, timeout=INF_TIMEOUT)
                if response and response.get("accepted", False):
                    # If any WebUI accepts, return success
                    return web.json_response(
                        {
                            "status": "ok",
                            "accepted": True,
                            "message": response.get("message", ""),
                        }
                    )
                responses.append(response)
            except Exception as e:
                log.error(f"Error sending to WebUI {manager.client_id}: {e}")
                responses.append(None)

        # If we get here, either all rejected or errored
        return web.json_response(
            {
                "status": "ok",
                "accepted": False,
                "message": "Workflow was rejected by all WebUIs",
            }
        )

    except Exception as e:
        log.error(f"Error sending workflow: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.post("/uiapi/clear_nonexistant_models")
async def uiapi_clear_nonexistant_models(request):
    """Clear widget values for models that don't exist on disk"""
    print()
    log.info("-> /uiapi/clear_nonexistant_models")
    webui = WebuiManager.get_main_webui()
    assert webui is not None

    try:
        workflow = await webui.wsend("get_workflow", buffered=True)
        nodes = workflow["workflow"]

        fields_to_clear = []

        # Analyze each node's widgets for non-existent models
        for node in nodes:
            if isinstance(node, dict) and "widgets_values" in node:
                for idx, value in enumerate(node["widgets_values"]):
                    if isinstance(value, str) and any(
                        value.endswith(ext) for ext in [".safetensors", ".ckpt", ".bin"]
                    ):
                        if not model_defs.has_model(value):
                            # Format: [node_id.widget_index, ""]
                            fields_to_clear.append([f"{node['id']}.{idx}", ""])

        if not fields_to_clear:
            return web.json_response(
                {
                    "status": "ok",
                    "message": "No non-existent models found to clear",
                    "cleared_count": 0,
                }
            )

        # Send set_fields request to clear the values
        request = await webui_manager.send(
            "set_fields", {"fields": fields_to_clear}, buffered=True
        )

        # Wait for response
        response = await webui_manager.wait(request)

        return web.json_response(
            {
                "status": "ok",
                "message": f"Cleared {len(fields_to_clear)} non-existent model values",
                "cleared_count": len(fields_to_clear),
                "cleared_fields": fields_to_clear,
                "set_fields_response": response,
            }
        )

    except Exception as e:
        log.error(f"Error clearing non-existent models: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.get("/uiapi/mcp_status")
async def uiapi_mcp_status(request):
    """Check current MCP installation status across all supported targets.

    Inspects Claude Desktop and Claude Code config files to determine whether
    the comfyui MCP server entry is present, without modifying anything.
    """
    try:
        import platform as plat

        plugin_path = str(Path(__file__).resolve().parent)
        system = plat.system()

        # Build target configs based on detected platform
        if system == "Linux":
            target_configs = {
                "claude_desktop": Path("~/.config/Claude/claude_desktop_config.json").expanduser(),
                "claude_code": Path("~/.claude.json").expanduser(),
            }
        elif system == "Darwin":
            target_configs = {
                "claude_desktop": Path("~/Library/Application Support/Claude/claude_desktop_config.json").expanduser(),
                "claude_code": Path("~/.claude.json").expanduser(),
            }
        elif system == "Windows":
            appdata = os.environ.get("APPDATA", "")
            target_configs = {
                "claude_desktop": Path(appdata) / "Claude" / "claude_desktop_config.json",
                "claude_code": Path("~/.claude.json").expanduser(),
            }
        else:
            target_configs = {
                "claude_code": Path("~/.claude.json").expanduser(),
            }

        targets = {}
        for name, config_path in target_configs.items():
            config_exists = config_path.exists()
            installed = False
            if config_exists:
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    installed = "comfyui" in config.get("mcpServers", {})
                except (json.JSONDecodeError, IOError):
                    pass
            targets[name] = {
                "path": str(config_path),
                "config_exists": config_exists,
                "installed": installed,
            }

        log.debug("-> /uiapi/mcp_status")
        return web.json_response({
            "status": "ok",
            "plugin_path": plugin_path,
            "platform": system,
            "targets": targets,
        })
    except Exception as e:
        log.error(f"Error checking MCP status: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.post("/uiapi/install_mcp")
async def uiapi_install_mcp(request):
    """Write MCP server config into Claude Desktop and/or Claude Code config files.

    Merges the comfyui MCP server entry into each target's JSON config,
    creating parent directories and the file itself if they don't exist.
    Accepts an optional JSON body with {targets: ["claude_desktop", "claude_code"]}
    to limit which configs are written (defaults to all detected targets).
    """
    try:
        import platform as plat

        # Parse optional request body for target filtering
        try:
            body = await request.json()
        except Exception:
            body = {}
        requested_targets = body.get("targets", None)

        plugin_path = str(Path(__file__).resolve().parent)

        # Detect ComfyUI address from the running PromptServer instance
        address = getattr(PromptServer.instance, "address", "127.0.0.1")
        port = getattr(PromptServer.instance, "port", 8188)
        comfyui_address = f"{address}:{port}"

        # The MCP server entry we want to install
        mcp_entry = {
            "command": "uvx",
            "args": ["--from", plugin_path, "comfyui-mcp-uiapi"],
            "env": {"COMFYUI_ADDRESS": comfyui_address},
        }

        system = plat.system()

        # Build target configs based on detected platform (same logic as mcp_status)
        if system == "Linux":
            target_configs = {
                "claude_desktop": Path("~/.config/Claude/claude_desktop_config.json").expanduser(),
                "claude_code": Path("~/.claude.json").expanduser(),
            }
        elif system == "Darwin":
            target_configs = {
                "claude_desktop": Path("~/Library/Application Support/Claude/claude_desktop_config.json").expanduser(),
                "claude_code": Path("~/.claude.json").expanduser(),
            }
        elif system == "Windows":
            appdata = os.environ.get("APPDATA", "")
            target_configs = {
                "claude_desktop": Path(appdata) / "Claude" / "claude_desktop_config.json",
                "claude_code": Path("~/.claude.json").expanduser(),
            }
        else:
            target_configs = {
                "claude_code": Path("~/.claude.json").expanduser(),
            }

        # Filter to only requested targets if specified
        if requested_targets is not None:
            target_configs = {
                k: v for k, v in target_configs.items() if k in requested_targets
            }

        results = {}
        for name, config_path in target_configs.items():
            try:
                # Read existing config or start fresh
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = json.load(f)
                else:
                    config = {}

                # Check if already installed with the exact same config
                existing_entry = config.get("mcpServers", {}).get("comfyui")
                if existing_entry == mcp_entry:
                    results[name] = {"status": "already_installed", "path": str(config_path)}
                    continue

                # Determine if this is a fresh install or an update
                result_status = "updated" if existing_entry is not None else "installed"

                # Merge the MCP entry into the config
                if "mcpServers" not in config:
                    config["mcpServers"] = {}
                config["mcpServers"]["comfyui"] = mcp_entry

                # Ensure parent directories exist
                config_path.parent.mkdir(parents=True, exist_ok=True)

                # Write back the config
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

                results[name] = {"status": result_status, "path": str(config_path)}
                log.info(f"MCP config {result_status} for {name} at {config_path}")

            except Exception as target_err:
                log.error(f"Error installing MCP for target {name}: {target_err}")
                results[name] = {
                    "status": "error",
                    "path": str(config_path),
                    "error": str(target_err),
                }

        log.debug("-> /uiapi/install_mcp")
        return web.json_response({
            "status": "ok",
            "plugin_path": plugin_path,
            "comfyui_address": comfyui_address,
            "results": results,
        })
    except Exception as e:
        log.error(f"Error installing MCP config: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)
