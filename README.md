# Chatbot RAG

Chatbot RAG en español que recupera fragmentos de una colección de Qdrant y
genera respuestas con un servidor vLLM compatible con la API de OpenAI. Se
puede usar desde la terminal o desde una interfaz web Django.

## Requisitos

- Python 3.10 o posterior.
- Qdrant accesible desde la máquina (por defecto, `127.0.0.1:6333`).
- Un servidor vLLM con el endpoint `/v1/chat/completions`.
- Una colección Qdrant ya creada y cargada con embeddings compatibles con
  `BAAI/bge-m3` y los metadatos de retrieval configurados.

## Instalación

Desde la raíz del repositorio, crea y activa un entorno virtual. El archivo
`requirements.txt` contiene las dependencias del chatbot RAG; si también vas a
usar Django, instala además `web/requirements.txt`.

```bash
python -m venv .venv
```

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r web\requirements.txt  # solo para la interfaz web
```

En macOS o Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -r web/requirements.txt  # solo para la interfaz web
```

## Configuración

El archivo [`configs/chatbot_config.json`](configs/chatbot_config.json)
configura Qdrant, el modelo de embeddings, el número de resultados y las
combinaciones de tipo de documento y estrategia que se buscan.

Define las variables de entorno `VLLM_URL` y `VLLM_MODEL` para indicar la URL
base del servidor vLLM y el nombre del modelo servido. Por ejemplo, en
PowerShell:

```powershell
$env:VLLM_URL = "http://localhost:8000"
$env:VLLM_MODEL = "nombre-del-modelo"
```

La URL base no debe incluir `/v1/chat/completions`; el cliente añade esa ruta.
El modelo de embeddings debe coincidir con el utilizado para poblar la
colección Qdrant.

## Uso desde la terminal

Con Qdrant y vLLM disponibles y las variables de entorno definidas:

```bash
python scripts/run_chatbot.py
```

Escribe `salir`, `exit` o `quit` para terminar.

## Interfaz web

La aplicación Django está en `web/`. Además de `VLLM_URL` y `VLLM_MODEL`,
necesita `DJANGO_SECRET_KEY`; opcionalmente admite `DJANGO_DEBUG` y
`DJANGO_ALLOWED_HOSTS`.

En PowerShell, configura una clave local y prepara la base de datos:

```powershell
$env:DJANGO_SECRET_KEY = (python -c "import secrets; print(secrets.token_urlsafe(50))")
python web\manage.py migrate
python web\manage.py createsuperuser
python web\manage.py runserver
```

Abre `http://127.0.0.1:8000/` e inicia sesión con el usuario creado. No uses
`runserver` ni `DJANGO_DEBUG=1` como configuración de producción; define una
clave secreta y hosts permitidos adecuados para el despliegue.
