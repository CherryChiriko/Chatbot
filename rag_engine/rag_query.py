from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import signal
import functools
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

CHROMA_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "reference_docs"



# Add timeout handler for LLM calls
def timeout_handler(signum, frame):
    raise TimeoutError("LLM inference timeout exceeded")

def with_timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            except TimeoutError:
                logger.error(f"Timeout in {func.__name__} after {seconds}s")
                raise
        return wrapper
    return decorator

embedding = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://localhost:11434" 
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=embedding
)

model = OllamaLLM(
    model="llama3.2",
    base_url="http://localhost:11434"  
)

prompt = ChatPromptTemplate.from_template("""
Tu es un assistant technique bancaire professionnel expert et chaleureux.

RÈGLES :
1. Réponds TOUJOURS en français.
2. Utilise UNIQUEMENT le contexte fourni pour répondre.
3. Si la réponse n'est pas dans le contexte, dis exactement : "Désolé, je ne trouve pas cette information dans ma documentation."
4. Si tu trouves une réponse partielle qui correspond à la question mais que tu détectes que l'utilisateur a posé plusieurs questions, réponds à la question partielle et indique que d'autres sujets sont en dehors de ton domaine. Exemple: "Je peux t'aider sur les cartes bancaires: [answer]. Cependant, les questions sur [l'autre sujet] sont en dehors de mon domaine."
5. Si tu ne peux pas répondre à la question à cause d'un problème technique ou de complexité, propose de contacter un conseiller humain.
6. Synthétise les informations de manière fluide. Ne fais pas de copier-coller brut.

Context:
{context}

Question:
{question}

Answer:
""")

chain = prompt | model

def save_to_log(session_id, question, answer, source):
    """Saves conversation details to a session-specific txt file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"./conversation_logs/chat_{session_id}.txt"
    
    log_entry = (
        f"--- {timestamp} ---\n"
        f"QUESTION: {question}\n"
        f"ANSWER: {answer}\n"
        f"SOURCE: {source}\n"
        f"{'-'*30}\n\n"
    )
    
    with open(filename, "a", encoding="utf-8") as f:
        f.write(log_entry)

def ask_rag(question: str, session_id: int) -> dict:
    try:
        # 1. Get the relevant documents 
        logger.info(f"Retrieving documents for: {question[:50]}...")
        raw_docs = vector_store.similarity_search(question, k=4)
        
        # 2a. VERSION PRIORITIZATION LOGIC
        # We use a dictionary to keep the 'best' version of each chunk
        final_docs_map = {}

        for doc in raw_docs:
            source = doc.metadata.get("source", "Unknown")
            version = doc.metadata.get("version", 1)
            # Create a unique key for the specific part of the file (e.g., page or row)
            # This ensures we don't just keep one chunk per file, but one 'newest' version of each chunk.
            chunk_key = f"{source}_{doc.metadata.get('page', doc.metadata.get('row', 'all'))}"

            if chunk_key not in final_docs_map:
                final_docs_map[chunk_key] = doc
            else:
                # If we already have this chunk, only replace it if this version is higher
                if version > final_docs_map[chunk_key].metadata.get("version", 1):
                    final_docs_map[chunk_key] = doc

        # 3. Finalize context (limit back down to 3-4 high-quality, newest docs)
        prioritized_docs = list(final_docs_map.values())[:4]
               
        # 4. Extract the context for the LLM (limit to first 3 docs to reduce tokens)
        docs = prioritized_docs[:3]
        sources = list(set([f"{d.metadata.get('source')} (V{d.metadata.get('version')})" for d in prioritized_docs]))

        context = "\n\n".join(d.page_content for d in docs)
        
        if not context.strip():
            return {"answer": "Désolé, je ne trouve pas cette information.", "source": None}

        # 5. Extract the unique sources
        sources = list(set([
            d.metadata.get("source", "Unknown").split("\\")[-1].split("/")[-1]
            for d in docs
        ]))

        if not context.strip():
            return {
                "answer": "Désolé, je ne trouve pas cette information dans ma documentation.",
                "source": None
            }

        # 6. Get the answer from the LLM with safety checks
        logger.info(f"Invoking LLM (context size: {len(context)} chars)...")
        answer = chain.invoke({
            "context": context,
            "question": question
        })

        logger.info(f"Question: {question}  \nAnswer: {answer}  \nSources: {sources}")  # For debugging and monitoring
        save_to_log(session_id, question, answer, ", ".join(sources))
        
        # 7. Return a dictionary so the Flask Bridge can pass it to Rasa
        return {
            "answer": answer,
            "source": ", ".join(sources)
        }
        
    except TimeoutError:
        logger.error("LLM timeout - question too complex or Ollama slow")
        return {
            "answer": "Désolé, je n'ai pas pu traiter votre question à temps. Pouvez-vous la reformuler plus simplement ?",
            "source": None
        }
    # except Exception as e:
    #     logger.error(f"Error in ask_rag: {str(e)}")
    #     save_to_log(session_id, question, "ERROR", str(e))
    #     return {
    #         "answer": "Je ne peux pas me connecter à la base de connaissances. Veuillez réessayer plus tard.",
    #         "source": None
    #     }
    except Exception as e:

        error_msg = traceback.format_exc()
        print("FULL ERROR:\n", error_msg)

        return {
            "answer": f"DEBUG ERROR:\n{str(e)}",
            "source": None
        }