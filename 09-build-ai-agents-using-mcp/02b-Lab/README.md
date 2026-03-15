install dependencies:

```bash
pip install -r requirements.txt
```

run client with server via stdio:

```bash
python client.py server.py
```

Set env vars for Hugging Face Router:

```bash
export HUGGINGFACEHUB_API_TOKEN="<your_hf_token>"
export HUGGINGFACE_MODEL="Qwen/Qwen3.5-397B-A17B:novita"
export HUGGINGFACE_BASE_URL="https://router.huggingface.co/v1"
```
