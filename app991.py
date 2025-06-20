# -*- coding: utf-8 -*-
import streamlit as st
import os
import openai
import weaviate
import re
import base64 # Import para Base64
import io # Necesario para trabajar con bytes en memoria
import urllib.parse # Para parsear la URL gs://
import json # Para parsear el JSON
import traceback # Para logs de error detallados

from sentence_transformers import SentenceTransformer
from weaviate.classes.init import Auth
# --- Importaciones de Google Cloud ---
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account

# --- MUEVE st.set_page_config AQUÍ PRIMERO ---
st.set_page_config(page_title="Chat con NorIA", page_icon="🤖", layout="wide")

# --- CONSTANTES ---
MODEL_NAME = "intfloat/multilingual-e5-large"
WEAVIATE_CLASS_NAME_DEFAULT = "Flujo_Caja_Mer_limpio2"
LOGO_IMAGE_PATH = "logo.png" 
LOGO_WIDTH_PX = 200
MAX_HISTORY_LENGTH = 15

# --- Funciones (Definiciones) ---

def image_to_base64(path):
    """Convierte un archivo de imagen local a una cadena Base64 Data URI."""
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        if path.lower().endswith(".png"): format = "png"
        elif path.lower().endswith((".jpg", ".jpeg")): format = "jpeg"
        elif path.lower().endswith(".gif"): format = "gif"
        elif path.lower().endswith(".svg"): format = "svg+xml"
        else: format = "octet-stream"
        return f"data:image/{format};base64,{encoded_string}"
    except FileNotFoundError:
        # Mostrar advertencia en UI si el logo no se encuentra
        st.warning(f"Advertencia: Archivo de logo no encontrado en '{path}'. No se mostrará el logo.")
        return None
    except Exception as e:
        st.error(f"Error al procesar el archivo de logo '{path}': {e}")
        return None

@st.cache_resource
def load_embedding_model(model_name: str):
    """
    Carga el modelo SentenceTransformer y lo cachea.
    Esta versión es silenciosa para el usuario (sin st.write/st.success).
    """
    # Mensajes de log para la consola/servidor (no para la UI de Streamlit)
    print(f"--- INFO: Intentando cargar modelo de embeddings '{model_name}' (cache check)...")
    try:
        model = SentenceTransformer(model_name)
        print(f"--- INFO: Modelo '{model_name}' cargado exitosamente en memoria (o recuperado de caché).")
        # NO st.write ni st.success aquí para mantenerlo silencioso
        return model
    except Exception as e:
        # SI mostrar error en UI si falla, porque la app no puede continuar
        print(f"--- ERROR FATAL: Cargando modelo '{model_name}' ---")
        print(traceback.format_exc())
        print(f"--- FIN ERROR ---")
        st.error(f"❌ Error fatal al cargar el modelo de embeddings '{model_name}'. La aplicación no puede iniciarse.")
        st.error(f"Detalle técnico: {e}")
        st.stop() # Detiene la ejecución de la app Streamlit
        return None

@st.cache_data(ttl=3600)
def download_blob_as_bytes(bucket_name, source_blob_name):
    """
    Descarga un blob de GCS como bytes usando las credenciales
    configuradas en st.secrets["gcp_service_account"]. Cacheado.
    """
    print(f"--- FUNC ENTER (cache check): download_blob_as_bytes(gs://{bucket_name}/{source_blob_name})")
    result = None
    if not bucket_name or not source_blob_name:
        print(f"---> ERROR FUNC ARGS: Bucket ('{bucket_name}') o Blob ('{source_blob_name}') inválido.")
        return result

    storage_client = None
    try:
        if "gcp_service_account" not in st.secrets:
            error_message = "Config Error: Credenciales 'gcp_service_account' no encontradas en st.secrets."
            print(f"---> ERROR FATAL FUNC: {error_message}")
            # No mostrar error en UI por descarga fallida de imagen, solo advertencia donde se use
            return result

        credentials_info = st.secrets["gcp_service_account"]
        required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email"]
        if not all(key in credentials_info and credentials_info[key] for key in required_keys):
            error_message = "Config Error: Credenciales 'gcp_service_account' incompletas en st.secrets."
            print(f"---> ERROR FATAL FUNC: {error_message}")
            return result

        # print(f"---> Creando credenciales GCS desde st.secrets...") # Log menos verboso
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        project_id = credentials_info.get("project_id")

        # print(f"---> Inicializando GCS Client (Project: {project_id})...") # Log menos verboso
        storage_client = storage.Client(credentials=credentials, project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        # print(f"---> Descargando gs://{bucket_name}/{source_blob_name}...") # Log menos verboso
        content = blob.download_as_bytes(timeout=60.0)
        print(f"---> Descarga completa! gs://{bucket_name}/{source_blob_name} ({len(content)} bytes).")
        result = content

    except NotFound:
        print(f"---> WARN FUNC: NotFound - gs://{bucket_name}/{source_blob_name} no existe.")
        result = None
    except Exception as e:
        print(f"---> ERROR FUNC: {type(e).__name__} - Error GCS descargando 'gs://{bucket_name}/{source_blob_name}':")
        print(traceback.format_exc())
        # No mostrar st.error aquí, devolver None y manejarlo donde se llame
        result = None

    # print(f"--- FUNC EXIT: download_blob_as_bytes. Devolviendo: {type(result)}") # Log menos verboso
    return result

def parse_gs_uri(gs_uri):
    """Parsea una URI gs:// y devuelve (bucket_name, object_path)."""
    if not gs_uri or not gs_uri.startswith("gs://"):
        # print(f"--- WARN: URI no válida (no empieza con gs://): {gs_uri}") # Menos verboso
        return None, None
    try:
        parsed = urllib.parse.urlparse(gs_uri)
        if parsed.scheme == "gs":
            bucket_name = parsed.netloc
            object_path = parsed.path.lstrip('/')
            if not bucket_name or not object_path:
                # print(f"--- WARN: URI inválida (falta bucket u objeto): {gs_uri}") # Menos verboso
                return None, None
            return bucket_name, object_path
        else:
            # print(f"--- WARN: URI no tiene esquema 'gs': {gs_uri}") # Menos verboso
            return None, None
    except Exception as e:
        print(f"--- ERROR: Parseando URI {gs_uri}: {e}")
        return None, None

def get_query_embedding(text):
    """Genera el embedding para una consulta usando el modelo cacheado."""
    query_with_prefix = "query: " + text
    return embedding_model.encode(query_with_prefix).tolist()

def retrieve_similar_chunks(query, k=5):
    """Recupera chunks de Weaviate basados en la similitud vectorial."""
    print(f"--- INFO: Generando embedding para la consulta: '{query[:50]}...'")
    query_vector = get_query_embedding(query)
    print(f"--- INFO: Buscando {k} chunks similares en Weaviate...")
    try:
        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=k,
            return_properties=[
                "text", "page_number", "source_pdf",
                "chunk_index_on_page", "image_urls"
            ]
        )
        context = []
        print(f"--- INFO: Weaviate devolvió {len(results.objects)} objetos.")
        for obj in results.objects:
            properties = obj.properties
            context.append({
                "text": properties.get("text", ""),
                "page_number": properties.get("page_number", -1),
                "source": properties.get("source_pdf", "N/A"),
                "chunk_index": properties.get("chunk_index_on_page", -1),
                "image_urls": properties.get("image_urls", []) or []
            })
        return context
    except Exception as e:
        print(f"--- ERROR: Durante la búsqueda en Weaviate ---")
        print(traceback.format_exc())
        st.error(f"❌ Error durante la búsqueda en Weaviate: {e}")
        return []

def remove_duplicate_chunks(chunks):
    """Elimina chunks duplicados basados en página y texto exacto."""
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        key = (chunk.get("page_number", -1), chunk.get("text", "").strip())
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    return unique_chunks

def group_chunks_by_page(chunks):
    """Agrupa chunks por número de página, recopilando textos y URLs de imagen únicas."""
    grouped = {}
    for chunk in chunks:
        page = chunk.get("page_number", -1)
        if page < 0: continue
        if page not in grouped:
            grouped[page] = {
                "texts": set(),
                "image_urls": set(chunk.get("image_urls", []) or [])
            }
        grouped[page]["texts"].add(chunk.get("text", "").strip())
        grouped[page]["image_urls"].update(chunk.get("image_urls", []) or [])
    final_grouped = {}
    for page, data in grouped.items():
        final_grouped[page] = {
            "texts": sorted(list(data["texts"])),
            "image_urls": sorted(list(data["image_urls"]))
        }
    return final_grouped

def generate_response(query, context):
    """Genera respuesta con OpenAI usando el contexto y extrae las páginas citadas."""
    if not context:
        return "No pude encontrar información relevante en el documento para responder a tu pregunta.", []

    context_text = "\n\n".join(
        f"[Fuente: {c.get('source', 'N/A')} - Página {c['page_number']}]: {c['text']}"
        for c in context
    )
    prompt = f"""Eres un asistente virtual experto llamado NorIA. Tu tarea es responder preguntas basándote EXCLUSIVAMENTE en el contexto proporcionado de un manual técnico.

INSTRUCCIONES IMPORTANTES:
1.  BASA TU RESPUESTA ÚNICAMENTE EN EL SIGUIENTE CONTEXTO. No inventes información ni uses conocimiento externo.
2.  Si la respuesta se encuentra en el contexto, respóndela de forma clara y organízala visualmente, si es posible, con viñetas.
3.  Si la respuesta NO se encuentra en el contexto, di EXACTAMENTE: "No encuentro información sobre eso en el contexto proporcionado." No intentes adivinar.
4.  Al final de tu respuesta (si encontraste información), AÑADE UNA LÍNEA SEPARADA que comience con "PÁGINAS CITADAS:" seguida de los números de página del contexto que utilizaste directamente para formar tu respuesta, separados por comas. Ejemplo: "PÁGINAS CITADAS: 15, 23".
5.  Si dijiste "No encuentro información...", entonces usa "PÁGINAS CITADAS: N/A".

CONTEXTO PROPORCIONADO:
---
{context_text}
---

PREGUNTA DEL USUARIO: {query}

RESPUESTA DE NorIA:"""

    print(f"--- INFO: Llamando a OpenAI con un prompt de longitud ~{len(prompt)}...")
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        response_text = response.choices[0].message.content
        print(f"--- INFO: Respuesta cruda de OpenAI recibida.")
    except Exception as e:
        print(f"--- ERROR: Llamando a la API de OpenAI ---")
        print(traceback.format_exc())
        st.error(f"❌ Error al llamar a la API de OpenAI: {e}")
        return "Hubo un error interno al intentar generar la respuesta.", []

    final_response = response_text.strip()
    used_pages_str = "N/A"
    used_pages = []
    match = re.search(r"\nPÁGINAS CITADAS:\s*(.*)$", response_text, re.IGNORECASE | re.MULTILINE)
    if match:
        used_pages_str = match.group(1).strip()
        final_response = response_text[:match.start()].strip()
        print(f"--- INFO: Páginas citadas encontradas por regex: '{used_pages_str}'")
        if used_pages_str.upper() != "N/A" and used_pages_str:
            try:
                used_pages = [int(p.strip()) for p in used_pages_str.split(',') if p.strip().isdigit()]
            except ValueError:
                print(f"--- WARN: No se pudieron parsear los números de página citados: '{used_pages_str}'")
                used_pages = []
        else:
             print(f"--- INFO: Páginas citadas marcadas como N/A o vacías.")
    else:
         print(f"--- WARN: No se encontró la línea 'PÁGINAS CITADAS:' en la respuesta.")

    print(f"--- INFO: Respuesta final procesada: '{final_response[:100]}...'")
    print(f"--- INFO: Páginas parseadas como usadas: {used_pages}")
    used_chunks_from_context = [
        c for c in context if c.get("page_number", -1) in used_pages
    ]
    unique_used_chunks_for_display = remove_duplicate_chunks(used_chunks_from_context)
    print(f"--- INFO: Devolviendo {len(unique_used_chunks_for_display)} chunks únicos marcados como usados.")
    return final_response, unique_used_chunks_for_display


# --- 1. Configuración y Conexiones (Ejecución Principal) ---

# Cargar Variables de Entorno
openai_api_key = os.environ.get("OPENAI_API_KEY")
weaviate_url = os.environ.get("WEAVIATE_URL")
weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")
weaviate_class_name = os.environ.get("WEAVIATE_CLASS_NAME", WEAVIATE_CLASS_NAME_DEFAULT)

# Validar variables de entorno críticas
if not openai_api_key:
    st.error("❌ Error Config: 'OPENAI_API_KEY' no configurada.")
    st.stop()
if not weaviate_url or not weaviate_api_key:
    st.error("❌ Error Config: 'WEAVIATE_URL' y/o 'WEAVIATE_API_KEY' no configuradas.")
    st.stop()
if not weaviate_class_name:
    st.error("❌ Error Config: Se necesita 'WEAVIATE_CLASS_NAME'.")
    st.stop()

# Configurar API Key de OpenAI
openai.api_key = openai_api_key

# --- Cargar el modelo (llamada a función cacheada, ahora silenciosa) ---
embedding_model = load_embedding_model(MODEL_NAME)
if embedding_model is None: # Doble check por si st.stop() fallara
     st.error("Fallo crítico en la carga del modelo. La aplicación se detendrá.")
     st.stop()

# --- Conectar a Weaviate ---
try:
    print(f"🔌 Conectando a Weaviate en {weaviate_url}...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )
    client.is_ready()
    print(f"✅ Conectado a Weaviate. Obteniendo colección '{weaviate_class_name}'...")
    collection = client.collections.get(weaviate_class_name)
    print(f"✅ Colección '{weaviate_class_name}' obtenida.")
except Exception as e:
    print(f"--- ERROR FATAL: Conectando a Weaviate o obteniendo colección ---")
    print(traceback.format_exc())
    st.error(f"❌ Error conectando a Weaviate ('{weaviate_class_name}'): {e}")
    st.stop()


# --- 2. Streamlit UI (Inicio) ---
# st.set_page_config ya se llamó al principio

# --- Logo Banner ---
logo_base64 = image_to_base64(LOGO_IMAGE_PATH)
if logo_base64:
    col1, col2, col3 = st.columns([1, 1, 6])
    with col2:
        st.image(logo_base64, width=LOGO_WIDTH_PX)
else:
    # Fallback si no hay logo
    st.markdown("<h1 style='text-align: center;'>Chat con NorIA 🤖</h1>", unsafe_allow_html=True)

# --- Colores (ajusta si es necesario) ---
color_azul = "#00205B"  # Un azul oscuro tipo corporativo
color_amarillo = "#EAAA00" # Un amarillo dorado/mostaza

# --- Título con colores ---
st.markdown(f"""
<h1 style='text-align: center;'>
    <span style='color: {color_azul};'>Chat con Nor</span><span style='color: {color_amarillo};'>IA</span> 🤖
</h1>
""", unsafe_allow_html=True)

st.write(f"Pregúntale a ChatNori")


# --- Inicializar historial de chat ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Limitar tamaño del historial (opcional) ---
if MAX_HISTORY_LENGTH and len(st.session_state.chat_history) > MAX_HISTORY_LENGTH * 2:
    print(f"--- INFO: Historial excede {MAX_HISTORY_LENGTH} pares. Truncando...")
    st.session_state.chat_history = st.session_state.chat_history[-(MAX_HISTORY_LENGTH * 2):]
    print(f"--- INFO: Historial truncado a {len(st.session_state.chat_history)} mensajes.")

# --- Mostrar historial de chat existente ---
for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "simplified_sources" in msg and msg["simplified_sources"]:
            st.markdown("---")
            st.markdown("<span style='font-size: 0.9em; color: grey;'>Fuentes consultadas para esta respuesta:</span>", unsafe_allow_html=True)

            grouped_simplified_sources = {}
            source_doc_name = "N/A"
            for source_info in msg["simplified_sources"]:
                page = source_info.get("page_number", -1)
                if page < 0: continue
                if source_doc_name == "N/A" and source_info.get("source", "N/A") != "N/A":
                    source_doc_name = source_info["source"]
                if page not in grouped_simplified_sources:
                    grouped_simplified_sources[page] = {
                        "image_urls": set(source_info.get("image_urls", []) or [])
                    }
                else:
                    grouped_simplified_sources[page]["image_urls"].update(source_info.get("image_urls", []) or [])

            for page_num, data in sorted(grouped_simplified_sources.items()):
                 with st.expander(f"📄 Página {page_num} (Doc: {source_doc_name})"):
                    if data.get("image_urls"):
                        for img_uri in sorted(list(data["image_urls"])):
                            img_bucket, img_object_path = parse_gs_uri(img_uri)
                            if img_bucket and img_object_path:
                                image_bytes = download_blob_as_bytes(img_bucket, img_object_path)
                                if image_bytes:
                                    st.image(image_bytes, caption=f"Imagen: {img_object_path}", use_column_width='auto')
                                else:
                                    # Advertencia sutil si la descarga falla
                                    st.caption(f"⚠️ No se pudo cargar imagen: `{img_object_path}`")
                            # else: # No mostrar advertencia por URI inválida en historial viejo
                            #     st.caption(f"⚠️ URL de imagen inválida: `{img_uri}`")
                    else:
                         st.caption("_No se asociaron imágenes a esta página en las fuentes._")

# --- Input del usuario ---
user_input = st.chat_input("Escribe tu pregunta a chat Nori...")

if user_input:
    # Añadir pregunta del usuario al historial
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Procesar y mostrar respuesta del asistente
    with st.chat_message("assistant"):
        with st.status("Consultando a NorIA...", expanded=False) as status:
            status.write("Analizando tu pregunta...")
            context_chunks = retrieve_similar_chunks(user_input)

            if not context_chunks:
                status.update(label="No se encontró información relevante.", state="warning", expanded=False)
                respuesta = "No pude encontrar información relevante en el manual para responder a tu pregunta."
                used_chunks_for_display = []
            else:
                status.write(f"Encontrados {len(context_chunks)} fragmentos. Generando respuesta...")
                respuesta, used_chunks_for_display = generate_response(user_input, context_chunks)
                status.update(label="Respuesta lista.", state="complete", expanded=False)

        # Mostrar respuesta principal
        st.markdown(respuesta)

        # Mostrar fuentes detalladas (con texto) para la respuesta ACTUAL
        if used_chunks_for_display:
            st.markdown("---")
            st.markdown("<span style='font-size: 0.9em; color: grey;'>Fuentes consultadas para esta respuesta:</span>", unsafe_allow_html=True)

            grouped_sources_for_display = group_chunks_by_page(used_chunks_for_display)
            source_doc_name = used_chunks_for_display[0].get('source', 'N/A') if used_chunks_for_display else "N/A"

            for page_num, data in sorted(grouped_sources_for_display.items()):
                with st.expander(f"📄 Página {page_num} (Doc: {source_doc_name})"):
                    if data.get("texts"):
                         st.markdown("**Contexto relevante:**")
                         for txt in data["texts"]:
                             st.markdown(f"> _{txt}_")
                    else:
                        st.caption("_No hay texto asociado directamente a esta página en las fuentes._")

                    if data.get("image_urls"):
                        st.markdown("**Imágenes:**")
                        for img_uri in data["image_urls"]:
                            img_bucket, img_object_path = parse_gs_uri(img_uri)
                            if img_bucket and img_object_path:
                                image_bytes = download_blob_as_bytes(img_bucket, img_object_path)
                                if image_bytes:
                                    st.image(image_bytes, caption=f"Imagen: {img_object_path}", use_column_width='auto')
                                else:
                                    st.warning(f"⚠️ No se pudo cargar imagen: `{img_object_path}`") # Advertencia más visible para imágenes actuales
                            else:
                                st.warning(f"⚠️ URL de imagen inválida: `{img_uri}`")

        # Crear lista simplificada para guardar en historial
        simplified_sources_to_save = []
        if used_chunks_for_display:
            processed_keys = set()
            for chunk in used_chunks_for_display:
                page = chunk.get("page_number", -1)
                source = chunk.get("source", "N/A")
                key = (page, source)
                if key not in processed_keys:
                     # Recopilar URLs únicas para esta página/fuente de todos los chunks usados
                     urls_for_key = list(set(url for c in used_chunks_for_display if c.get("page_number") == page and c.get("source") == source for url in c.get("image_urls", []) or []))
                     simplified_sources_to_save.append({
                         "page_number": page,
                         "source": source,
                         "image_urls": sorted(urls_for_key) # Guardar URLs ordenadas
                     })
                     processed_keys.add(key)

        # Añadir respuesta al historial con fuentes SIMPLIFICADAS
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": respuesta,
            "simplified_sources": simplified_sources_to_save
        })

        # Opcional: Re-aplicar límite de historial
        if MAX_HISTORY_LENGTH and len(st.session_state.chat_history) > MAX_HISTORY_LENGTH * 2:
            st.session_state.chat_history = st.session_state.chat_history[-(MAX_HISTORY_LENGTH * 2):]

        # st.rerun() # Descomentar solo si la actualización no es fluida

# --- Fin del Script ---

# Mensaje final o pie de página (opcional)
# st.markdown("---")
# st.caption("NorIA - Asistente virtual v1.0")
