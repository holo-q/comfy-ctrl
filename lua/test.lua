-- test.lua

local comfy = require("comfy_client")

-- 1) Instantiate with colon, not dot, so `self` is passed in correctly:
--    this sets client.server_address = "127.0.0.1:8188"
local client = comfy:new("127.0.0.1:8188")

-- 2) Connect (will error if it can’t reach ComfyUI)
print("➤ Connecting to ComfyUI…")
local ok, err = pcall(function() return client:ensure_connection() end)
if not ok then
  io.stderr:write("✖ Failed to connect: ", tostring(err), "\n")
  os.exit(1)
end
print("✔ Connected!")

-- 3) Fetch the workflow JSON
print("➤ Fetching workflow…")
local wf = client:json_get("/uiapi/get_workflow")
local nodes = wf.response or wf
print("Nodes in workflow:")
for id,node in pairs(nodes) do
  local title = node._meta and node._meta.title or "(no title)"
  print(" •", id, title)
end

-- 4) Tweak a field (e.g. set ScaleImage.scale → 0.5)
print("➤ Setting ScaleImage.scale → 0.5")
client:set("ScaleImage.scale", 0.5)

-- 5) Execute and grab the raw PNG bytes
print("➤ Executing…")
local img_bytes = client:execute()

-- 6) Write the image out
local out = assert(io.open("out.png","wb"))
out:write(img_bytes)
out:close()
print("✔ Saved result to out.png")
