from fal_serverless import isolated, cached

MODEL_URL = "https://huggingface.co/TheBloke/samantha-1.1-llama-13B-GGML/resolve/main/samantha-1.1-llama-13b.ggmlv3.q4_0.bin"
MODEL_PATH = "/data/models/samantha-1.1-llama-13b.ggmlv3.q4_0.bin"

@isolated(
    kind="conda",
    env_yml="llama_cpp_env.yml",
    machine_type="M",
)
def download_model():
    print("---> This is download_model()")
    import os

    if not os.path.exists("/data/models"):
        os.system("mkdir /data/models")
    if not os.path.exists(MODEL_PATH):
        print("Downloading SAM model.")
        os.system(f"cd /data/models && wget {MODEL_URL}")

@isolated(
    kind="conda",
    env_yml="llama_cpp_env.yml",
    machine_type="GPU-T4",
    exposed_port=8080,
    keep_alive=30
)
def llama_server():
    import uvicorn
    from llama_cpp.server import app

    settings = app.Settings(model=MODEL_PATH, n_gpu_layers=96)

    server = app.create_app(settings=settings)
    uvicorn.run(server, host="0.0.0.0", port=8080)