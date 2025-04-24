# -*- coding: utf-8 -*-
import streamlit as st
import os
import openai
import weaviate
import re
import base64 # Import para Base64
import io # Necesario para trabajar con bytes en memoria
import urllib.parse # Para parsear la URL gs://
import json # Para parsear el JSON (aunque no se usa explícitamente aquí, puede ser útil)
import traceback # Para logs de error detallados

from sentence_transformers import SentenceTransformer
from weaviate.classes.init import Auth
# --- Importaciones de Google Cloud ---
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account

# --- CONSTANTES ---
# Nombre del modelo de embeddings a usar
MODEL_NAME = "intfloat/multilingual-e5-large"
# Nombre por defecto de la clase/colección en Weaviate si no se especifica en variable de entorno
WEAVIATE_CLASS_NAME_DEFAULT = "Flujo_Caja_Mer_limpio2"
# Ruta al archivo de logo local (asegúrate de que exista)
LOGO_IMAGE_PATH = "logo.png" # ¡¡CAMBIA ESTO A LA RUTA DE TU LOGO!!
LOGO_WIDTH_PX = 200 # Ancho del logo en píxeles
# Máximo número de PARES de mensajes (usuario + asistente) a mantener en el historial
# Poner un número grande o None para desactivar el límite
MAX_HISTORY_LENGTH = 15

# --- Función para convertir imagen local a Base64 (para el logo) ---
def image_to_base64(path):
    """Convierte un archivo de imagen local a una cadena Base64 Data URI."""
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        if path.lower().endswith(".png"): format = "png"
        elif path.lower().endswith((".jpg", ".jpeg")): format = "jpeg"
        elif path.lower().endswith(".gif"): format = "gif"
        elif path.lower().endswith(".svg"): format = "svg+xml"
        else: format = "octet-stream" # Fallback genérico
        return f"data:image/{format};base64,{encoded_string}"
    except FileNotFoundError:
        st.warning(f"Advertencia: Archivo de logo no encontrado en '{path}'. No se mostrará el logo.")
        return None
    except Exception as e:
        st.error(f"Error al procesar el archivo de logo '{path}': {e}")
        return None

# --- 1. Configuración y Conexiones ---

# Cargar Variables de Entorno
openai_api_key = os.environ.get("OPENAI_API_KEY")
weaviate_url = os.environ.get("WEAVIATE_URL")
weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")
weaviate_class_name = os.environ.get("WEAVIATE_CLASS_NAME", WEAVIATE_CLASS_NAME_DEFAULT)

# Validar variables de entorno críticas
if not openai_api_key:
    st.error("❌ Error: La variable de entorno 'OPENAI_API_KEY' no está configurada.")
    st.stop()
if not weaviate_url or not weaviate_api_key:
    st.error("❌ Error: 'WEAVIATE_URL' y/o 'WEAVIATE_API_KEY' no están configuradas.")
    st.stop()
if not weaviate_class_name:
    st.error("❌ Error: Se necesita el nombre de la clase Weaviate (configura 'WEAVIATE_CLASS_NAME').")
    st.stop()

# Configurar API Key de OpenAI
openai.api_key = openai_api_key

# --- OPTIMIZACIÓN: Función Cacheada para Cargar el Modelo de Embeddings ---
@st.cache_resource # Usar cache_resource para objetos grandes como modelos
def load_embedding_model(model_name: str):
    """
    Carga el modelo SentenceTransformer desde HuggingFace y lo cachea
    para evitar recargarlo en cada ejecución del script.
    """
    print(f"--- INFO: Intentando cargar modelo de embeddings '{model_name}'...")
    st.write(f"⏳ Cargando modelo de embeddings '{model_name}' (esto solo debería aparecer una vez por sesión)...")
    try:
        model = SentenceTransformer(model_name)
        print(f"--- INFO: Modelo '{model_name}' cargado exitosamente.")
        st.success(f"✅ Modelo de embeddings '{model_name}' listo.")
        return model
    except Exception as e:
        print(f"--- ERROR FATAL: Cargando modelo '{model_name}' ---")
        print(traceback.format_exc())
        print(f"--- FIN ERROR ---")
        st.error(f"❌ Error fatal al cargar el modelo de embeddings '{model_name}': {e}")
        st.error("La aplicación no puede continuar sin el modelo. Revisa los logs del servidor.")
        st.stop() # Detiene la ejecución de la app Streamlit
        return None

# --- Cargar el modelo llamando a la función cacheada ---
embedding_model = load_embedding_model(MODEL_NAME)

# Verificación post-carga (por si st.stop() tuviera problemas en algún entorno)
if embedding_model is None:
     st.error("Fallo crítico en la carga del modelo de embeddings después de la llamada cacheada. La aplicación se detendrá.")
     st.stop()

# --- Conectar a Weaviate ---
try:
    print(f"🔌 Conectando a Weaviate en {weaviate_url}...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        # Opcional: añadir headers si tu proveedor lo requiere
        # headers={ "X-OpenAI-Api-Key": openai_api_key }
    )
    client.is_ready() # Verificar conexión
    print(f"✅ Conectado a Weaviate. Obteniendo colección '{weaviate_class_name}'...")
    collection = client.collections.get(weaviate_class_name)
    print(f"✅ Colección '{weaviate_class_name}' obtenida.")
except Exception as e:
    print(f"--- ERROR FATAL: Conectando a Weaviate o obteniendo colección ---")
    print(traceback.format_exc())
    print(f"--- FIN ERROR ---")
    st.error(f"❌ Error conectando a Weaviate o obteniendo la colección '{weaviate_class_name}': {e}")
    st.error("Verifica la URL, API Key, nombre de la clase y la conectividad de red.")
    st.stop()

# --- 2. Funciones Auxiliares ---

# --- Funciones de GCS (con caché de datos) ---
@st.cache_data(ttl=3600) # Cachea los bytes de la imagen por 1 hora
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
            st.error(error_message)
            return result

        credentials_info = st.secrets["gcp_service_account"]
        required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email"]
        if not all(key in credentials_info and credentials_info[key] for key in required_keys):
            error_message = "Config Error: Credenciales 'gcp_service_account' incompletas en st.secrets."
            print(f"---> ERROR FATAL FUNC: {error_message}")
            st.error(error_message)
            return result

        print(f"---> Creando credenciales GCS desde st.secrets...")
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        project_id = credentials_info.get("project_id")

        print(f"---> Inicializando GCS Client (Project: {project_id})...")
        storage_client = storage.Client(credentials=credentials, project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        print(f"---> Descargando gs://{bucket_name}/{source_blob_name}...")
        # Timeout para evitar bloqueos indefinidos
        content = blob.download_as_bytes(timeout=60.0)
        print(f"---> Descarga completa! ({len(content)} bytes).")
        result = content

    except NotFound:
        print(f"---> EXCEPTION FUNC: NotFound - gs://{bucket_name}/{source_blob_name} no existe.")
        # No mostrar error en UI, solo log. La función que llama manejará el None.
        result = None
    except Exception as e:
        print(f"---> EXCEPTION FUNC: {type(e).__name__} - Error GCS descargando 'gs://{bucket_name}/{source_blob_name}':")
        print(traceback.format_exc())
        st.warning(f"⚠️ No se pudo descargar la imagen: {source_blob_name}. Error de GCS.")
        result = None

    print(f"--- FUNC EXIT: download_blob_as_bytes. Devolviendo: {type(result)}")
    return result

def parse_gs_uri(gs_uri):
    """Parsea una URI gs:// y devuelve (bucket_name, object_path)."""
    if not gs_uri or not gs_uri.startswith("gs://"):
        print(f"--- WARN: URI no válida (no empieza con gs://): {gs_uri}")
        return None, None
    try:
        parsed = urllib.parse.urlparse(gs_uri)
        if parsed.scheme == "gs":
            bucket_name = parsed.netloc
            object_path = parsed.path.lstrip('/')
            if not bucket_name or not object_path:
                print(f"--- WARN: URI inválida (falta bucket u objeto): {gs_uri}")
                return None, None
            return bucket_name, object_path
        else:
            print(f"--- WARN: URI no tiene esquema 'gs': {gs_uri}")
            return None, None
    except Exception as e:
        print(f"--- ERROR: Parseando URI {gs_uri}: {e}")
        return None, None

# --- Funciones de RAG (Recuperación y Generación Aumentada) ---
def get_query_embedding(text):
    """Genera el embedding para una consulta usando el modelo cacheado."""
    query_with_prefix = "query: " + text # Prefijo para modelos E5
    # Llama al modelo cargado y cacheado globalmente
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
                "text", # Texto del chunk
                "page_number", # Número de página
                "source_pdf", # Nombre del documento fuente
                "chunk_index_on_page", # Índice del chunk en la página (si existe)
                "image_urls" # URLs de imágenes asociadas (espera una lista)
            ]
            # Opcional: return_metadata=['distance'] para ver la similitud
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
                # Asegurarse de que image_urls sea siempre una lista
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
        # Usar strip() para ignorar espacios en blanco al inicio/final
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
        if page < 0: continue # Ignorar chunks sin página válida

        if page not in grouped:
            grouped[page] = {
                "texts": set(), # Usar set para evitar duplicados de texto fácilmente
                "image_urls": set(chunk.get("image_urls", []) or []) # Usar set para URLs únicas
            }
        # Añadir texto al set (ignora duplicados automáticamente)
        grouped[page]["texts"].add(chunk.get("text", "").strip())
        # Añadir URLs de imagen al set
        grouped[page]["image_urls"].update(chunk.get("image_urls", []) or [])

    # Convertir sets de nuevo a listas para consistencia (opcional)
    final_grouped = {}
    for page, data in grouped.items():
        final_grouped[page] = {
            "texts": sorted(list(data["texts"])), # Ordenar textos alfabéticamente
            "image_urls": sorted(list(data["image_urls"])) # Ordenar URLs
        }
    return final_grouped


def generate_response(query, context):
    """Genera respuesta con OpenAI usando el contexto y extrae las páginas citadas."""
    if not context:
        return "No pude encontrar información relevante en el documento para responder a tu pregunta.", []

    # Crear el texto del contexto para el prompt
    context_text = "\n\n".join(
        # Incluir fuente y página claramente para ayudar al LLM a citar
        f"[Fuente: {c.get('source', 'N/A')} - Página {c['page_number']}]: {c['text']}"
        for c in context
    )

    # Prompt mejorado para guiar al LLM
    prompt = f"""Eres un asistente virtual experto llamado NorIA. Tu tarea es responder preguntas basándote EXCLUSIVAMENTE en el contexto proporcionado de un manual técnico.

INSTRUCCIONES IMPORTANTES:
1.  BASA TU RESPUESTA ÚNICAMENTE EN EL SIGUIENTE CONTEXTO. No inventes información ni uses conocimiento externo.
2.  Si la respuesta se encuentra en el contexto, respóndela de forma clara y concisa.
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
            model="gpt-3.5-turbo", # O considera "gpt-4-turbo-preview" si tienes acceso y necesitas más capacidad
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1 # Baja temperatura para respuestas más basadas en hechos
        )
        response_text = response.choices[0].message.content
        print(f"--- INFO: Respuesta cruda de OpenAI recibida.")

    except Exception as e:
        print(f"--- ERROR: Llamando a la API de OpenAI ---")
        print(traceback.format_exc())
        st.error(f"❌ Error al llamar a la API de OpenAI: {e}")
        return "Hubo un error interno al intentar generar la respuesta.", []

    # Extracción de la respuesta y las páginas citadas
    final_response = response_text.strip()
    used_pages_str = "N/A"
    used_pages = [] # Lista de números de página (int)

    # Buscar la línea de páginas citadas al final
    match = re.search(r"\nPÁGINAS CITADAS:\s*(.*)$", response_text, re.IGNORECASE | re.MULTILINE)

    if match:
        used_pages_str = match.group(1).strip()
        # Quitar la línea de citas de la respuesta principal
        final_response = response_text[:match.start()].strip()
        print(f"--- INFO: Páginas citadas encontradas por regex: '{used_pages_str}'")
        if used_pages_str.upper() != "N/A" and used_pages_str:
            try:
                # Limpiar y convertir a enteros, ignorando no dígitos
                used_pages = [int(p.strip()) for p in used_pages_str.split(',') if p.strip().isdigit()]
            except ValueError:
                print(f"--- WARN: No se pudieron parsear los números de página citados: '{used_pages_str}'")
                used_pages = [] # Resetear si hay error
        else:
             print(f"--- INFO: Páginas citadas marcadas como N/A o vacías.")
    else:
         print(f"--- WARN: No se encontró la línea 'PÁGINAS CITADAS:' en la respuesta.")
         # Dejar la respuesta como está si no se encuentra el patrón

    print(f"--- INFO: Respuesta final procesada: '{final_response[:100]}...'")
    print(f"--- INFO: Páginas parseadas como usadas: {used_pages}")

    # Filtrar los chunks originales del contexto basado en las páginas citadas por el LLM
    # Solo devolveremos los chunks que el LLM dice haber usado
    used_chunks_from_context = [
        c for c in context if c.get("page_number", -1) in used_pages
    ]

    # Opcional: Eliminar duplicados entre los chunks usados antes de devolverlos
    unique_used_chunks_for_display = remove_duplicate_chunks(used_chunks_from_context)
    print(f"--- INFO: Devolviendo {len(unique_used_chunks_for_display)} chunks únicos marcados como usados.")

    return final_response, unique_used_chunks_for_display


# --- 3. Streamlit UI ---

st.set_page_config(page_title="Chat con NorIA", page_icon="🤖", layout="wide")

# --- Logo Banner ---
logo_base64 = image_to_base64(LOGO_IMAGE_PATH)
if logo_base64:
    # Usar columnas para controlar mejor la posición y tamaño
    col1, col2, col3 = st.columns([1, 4, 1]) # Ajusta las proporciones si es necesario
    with col1:
         st.write("") # Espacio a la izquierda
    with col2:
        st.image(logo_base64, width=LOGO_WIDTH_PX)
    with col3:
        st.write("") # Espacio a la derecha
else:
    # Si no hay logo, solo mostrar título centrado
    st.markdown("<h1 style='text-align: center;'>Chat con NorIA 🤖</h1>", unsafe_allow_html=True)

# Si hay logo, el título puede ir debajo o no, como prefieras
st.markdown("<h2 style='text-align: center;'>Asistente del Manual de Procedimientos</h2>", unsafe_allow_html=True)
st.markdown("---")


# --- Inicializar historial de chat en Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- OPTIMIZACIÓN: Limitar tamaño del historial (opcional) ---
if MAX_HISTORY_LENGTH and len(st.session_state.chat_history) > MAX_HISTORY_LENGTH * 2:
    print(f"--- INFO: Historial excede {MAX_HISTORY_LENGTH} pares. Truncando...")
    # Conservar solo los últimos N pares de mensajes
    st.session_state.chat_history = st.session_state.chat_history[-(MAX_HISTORY_LENGTH * 2):]
    print(f"--- INFO: Historial truncado a {len(st.session_state.chat_history)} mensajes.")

# --- Mostrar historial de chat existente ---
for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Mostrar fuentes simplificadas si existen (para mensajes antiguos)
        # 'simplified_sources' es la clave que usamos para guardar la info optimizada
        if msg["role"] == "assistant" and "simplified_sources" in msg and msg["simplified_sources"]:
            st.markdown("---") # Separador visual
            st.markdown("<span style='font-size: 0.9em; color: grey;'>Fuentes consultadas para esta respuesta:</span>", unsafe_allow_html=True)

            # Agrupar fuentes simplificadas por página para mostrarlas
            grouped_simplified_sources = {}
            source_doc_name = "N/A" # Intentar obtener el nombre del documento
            for source_info in msg["simplified_sources"]:
                page = source_info.get("page_number", -1)
                if page < 0: continue

                # Tomar el nombre del documento de la primera fuente que lo tenga
                if source_doc_name == "N/A" and source_info.get("source", "N/A") != "N/A":
                    source_doc_name = source_info["source"]

                if page not in grouped_simplified_sources:
                    grouped_simplified_sources[page] = {
                        "image_urls": set(source_info.get("image_urls", []) or [])
                        # Ya no tenemos el texto aquí
                    }
                else:
                    # Añadir URLs de imagen únicas
                    grouped_simplified_sources[page]["image_urls"].update(source_info.get("image_urls", []) or [])

            # Mostrar las fuentes agrupadas
            for page_num, data in sorted(grouped_simplified_sources.items()):
                 # Solo mostramos la referencia de página y las imágenes
                 with st.expander(f"📄 Página {page_num} (Doc: {source_doc_name})"):
                    if data.get("image_urls"):
                        # Código para mostrar imágenes (igual que antes)
                        for img_uri in sorted(list(data["image_urls"])): # Ordenar URLs
                            img_bucket, img_object_path = parse_gs_uri(img_uri)
                            if img_bucket and img_object_path:
                                image_bytes = download_blob_as_bytes(img_bucket, img_object_path)
                                if image_bytes:
                                    st.image(image_bytes, caption=f"Imagen: {img_object_path}", use_column_width='auto') # 'auto' puede ser mejor que 'True'
                                else:
                                    st.warning(f"⚠️ No se pudo cargar imagen: `{img_object_path}`")
                            else:
                                st.warning(f"⚠️ URL de imagen inválida: `{img_uri}`")
                    else:
                         st.markdown("_No se asociaron imágenes a esta página en las fuentes._")


# --- Input del usuario ---
user_input = st.chat_input("Escribe tu pregunta sobre el manual...")

if user_input:
    # Añadir pregunta del usuario al historial y mostrarla
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Procesar y mostrar respuesta del asistente
    with st.chat_message("assistant"):
        # Usar st.status para indicar progreso
        with st.status("Consultando a NorIA...", expanded=False) as status:
            status.write("Analizando tu pregunta...")
            # 1. Recuperar chunks relevantes (función existente)
            context_chunks = retrieve_similar_chunks(user_input)

            if not context_chunks:
                status.update(label="No se encontró información relevante.", state="warning")
                respuesta = "No pude encontrar información relevante en el manual para responder a tu pregunta."
                used_chunks_for_display = [] # Lista vacía
            else:
                status.write(f"Encontrados {len(context_chunks)} fragmentos relevantes. Generando respuesta...")
                # 2. Generar respuesta usando los chunks (función existente)
                #    Devuelve el texto de la respuesta y los chunks *realmente usados* por el LLM
                respuesta, used_chunks_for_display = generate_response(user_input, context_chunks)
                status.write("Respuesta generada.")

            # Marcar el status como completo
            status.update(label="Respuesta lista.", state="complete", expanded=False)

        # Mostrar la respuesta principal del asistente
        st.markdown(respuesta)

        # Mostrar las fuentes detalladas (con texto) SÓLO para esta respuesta actual
        if used_chunks_for_display:
            st.markdown("---") # Separador visual
            st.markdown("<span style='font-size: 0.9em; color: grey;'>Fuentes consultadas para esta respuesta:</span>", unsafe_allow_html=True)

            # Agrupar los chunks COMPLETOS usados para mostrarlos AHORA
            grouped_sources_for_display = group_chunks_by_page(used_chunks_for_display)
            source_doc_name = used_chunks_for_display[0].get('source', 'N/A') if used_chunks_for_display else "N/A"

            for page_num, data in sorted(grouped_sources_for_display.items()):
                with st.expander(f"📄 Página {page_num} (Doc: {source_doc_name})"):
                    # Mostrar textos relevantes de esta página
                    if data.get("texts"):
                         st.markdown("**Contexto relevante:**")
                         for txt in data["texts"]:
                             # Usar blockquote para el texto citado
                             st.markdown(f"> _{txt}_")
                    else:
                        st.markdown("_No hay texto asociado directamente a esta página en las fuentes._")

                    # Mostrar imágenes de esta página
                    if data.get("image_urls"):
                        st.markdown("**Imágenes:**")
                        for img_uri in data["image_urls"]: # Ya están ordenadas por group_chunks_by_page
                            img_bucket, img_object_path = parse_gs_uri(img_uri)
                            if img_bucket and img_object_path:
                                image_bytes = download_blob_as_bytes(img_bucket, img_object_path)
                                if image_bytes:
                                    st.image(image_bytes, caption=f"Imagen: {img_object_path}", use_column_width='auto')
                                else:
                                    st.warning(f"⚠️ No se pudo cargar imagen: `{img_object_path}`")
                            else:
                                st.warning(f"⚠️ URL de imagen inválida: `{img_uri}`")

        # --- OPTIMIZACIÓN: Crear lista simplificada ANTES de guardar en historial ---
        simplified_sources_to_save = []
        if used_chunks_for_display:
            # Usar un set para evitar duplicados de página/fuente/url al simplificar
            processed_keys = set()
            for chunk in used_chunks_for_display:
                page = chunk.get("page_number", -1)
                source = chunk.get("source", "N/A")
                key = (page, source) # Clave para agrupar urls por página/fuente
                if key not in processed_keys:
                     simplified_sources_to_save.append({
                         "page_number": page,
                         "source": source,
                         # Recopilar TODAS las urls de esta página/fuente de una vez
                         "image_urls": list(set(url for c in used_chunks_for_display if c.get("page_number") == page and c.get("source") == source for url in c.get("image_urls", []) or []))
                     })
                     processed_keys.add(key)


        # Añadir respuesta completa (texto + fuentes SIMPLIFICADAS) al historial
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": respuesta,
            "simplified_sources": simplified_sources_to_save # Guardar la lista optimizada
        })

        # Opcional: Re-aplicar límite de historial si se añadió un nuevo par
        if MAX_HISTORY_LENGTH and len(st.session_state.chat_history) > MAX_HISTORY_LENGTH * 2:
            st.session_state.chat_history = st.session_state.chat_history[-(MAX_HISTORY_LENGTH * 2):]

        # Forzar re-ejecución para actualizar la vista del historial inmediatamente
        # Puede ser útil si la actualización no es instantánea, pero usar con cuidado
        # st.rerun()

# Mensaje final o pie de página (opcional)
# st.markdown("---")
# st.caption("NorIA - Asistente virtual v1.0")