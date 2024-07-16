# HRAFL

Hybrid Resources Arrangement for Federated Leaning with LLM

# Environment

The development environment is list below:

- OS: Ubuntu 22.04.3 LTS
- CUDA: 12.1
- Python: 3.11

# How to run

## Simple FL example

The simple FL server is built on [fastapi](https://fastapi.tiangolo.com/).

Remember to install fastapi before start it.

```
pip install fastapi
```

Then, run the server in root under development mode:

```
fastapi dev simple_fl/server.py
```

After that, we can run client on different server:

```
python client.py --client_id client_123 --client_secret my_secret_key --train_script train_script/dry-run.py
```

The example used the demo train script `dry-run.py`.
