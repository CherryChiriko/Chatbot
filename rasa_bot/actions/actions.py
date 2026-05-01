import logging
import requests
import uuid
import time
from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def create_glpi_ticket(content: str) -> Optional[str]:
    """Call GLPI REST API to create a new ticket."""
    ticket_id = str(uuid.uuid4())[:8]
    mock_url = f"https://glpi.example.com/front/ticket.form.php?id={ticket_id}"
    
    logger.info(f"[MOCK] GLPI Ticket created: {mock_url}")
    return mock_url

# --- Actions ---

class ActionQueryRag(Action):
    def name(self) -> Text:
        return "action_query_rag"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get('text', "")
        session_id = tracker.sender_id
        current_failures = tracker.get_slot("rag_failure_count") or 0
        
        # --- GIBBERISH / SHORT MESSAGE FILTER ---
        # If it's pure gibberish (short + no vowels) or just random noise
        vowels = "aeiouyàâéèêëîïôûù"
        is_random = len(user_message) < 5 and not any(c in user_message.lower() for c in vowels)
        
        if is_random or not user_message.strip():
            dispatcher.utter_message(text="Désolé, je n'ai pas compris votre message. Pourriez-vous reformuler avec une phrase complète ?")
            return [] # We don't count gibberish as a RAG failure

        # 1. Check for max failures
        if current_failures >= 2:
            return [FollowupAction("action_transfer_to_human")]

        try:
            # Reduced timeout from 60s to 15s (60s is way too long for a user to wait)
            response = requests.post(
                "http://127.0.0.1:5006/ask", 
                json={"question": user_message, "session_id": session_id},
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                # Use .strip() to ensure we don't have empty strings that pass 'if answer'
                answer = data.get("answer", "").strip()
                source = data.get("source", "Document inconnu")

                # Define failure/technical phrases
                failure_phrases = ["pas disponible", "désolé", "en dehors de mon domaine", "pas trouvé"]
                technical_error_phrases = ["connexion", "erreur technique", "réessayer plus tard", "impossible de se connecter"]
                
                # Case A: Empty Answer
                if not answer:
                    dispatcher.utter_message(text="Je n'ai pas trouvé d'information à ce sujet dans mes documents.")
                    return [SlotSet("rag_failure_count", current_failures + 1)]

                # Case B: Technical error from the RAG script (NO buttons)
                if any(p in answer.lower() for p in technical_error_phrases):
                    dispatcher.utter_message(text=f"⚠️ {answer}")
                    return [] # Don't increment failure for server issues

                # Case C: "I don't know" answer (Functional failure)
                if any(p in answer.lower() for p in failure_phrases):
                    new_count = current_failures + 1
                    dispatcher.utter_message(text=f"{answer}\n\nPourriez-vous préciser votre demande ?")
                    return [SlotSet("rag_failure_count", new_count)]

                # Case D: SUCCESS → buttons
                dispatcher.utter_message(
                    text=answer,
                    buttons=[
                        {"title": "👍 Utile", "payload": "/affirm"},
                        {"title": "👎 Pas utile", "payload": "/inform_failure"}
                    ]
                )   
                return [SlotSet("last_source", source), SlotSet("rag_failure_count", 0)]

            else:
                dispatcher.utter_message(text="Le service de connaissances est momentanément indisponible (Erreur HTTP).")
                return []

        except Exception as e:
            # CRITICAL: Always return a list, never a dict!
            logger.error(f"Action Server Error: {e}")
            print(f"REAL ERROR: {e}") 
            dispatcher.utter_message(text="Je rencontre une difficulté pour accéder à ma base de données.")
            return []


class ActionHandleRagFailure(Action):
    """Increments failure count when user clicks 'Pas utile'"""
    def name(self) -> Text:
        return "action_handle_rag_failure"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        current_failures = tracker.get_slot("rag_failure_count") or 0
        new_count = current_failures + 1
        
        if new_count >= 2:
            return [SlotSet("rag_failure_count", new_count), FollowupAction("action_transfer_to_human")]
        
        dispatcher.utter_message(text="Je suis navré que cette réponse ne vous aide pas. Pourriez-vous reformuler ou préciser votre demande ?")
        return [SlotSet("rag_failure_count", new_count)]

class ActionTransferToHuman(Action):
    def name(self) -> Text:
        return "action_transfer_to_human"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get('text', "Besoin d'aide (échecs RAG répétés)")
        ticket_url = create_glpi_ticket(user_message)

        if ticket_url:
            dispatcher.utter_message(
                text=f"Je n'ai pas réussi à vous aider malgré mes tentatives. J'ai créé un ticket pour qu'un conseiller reprenne la main : {ticket_url}"
            )
            return [SlotSet("rag_failure_count", 0), SlotSet("glpi_ticket_url", ticket_url)]
        
        dispatcher.utter_message(text="Je vous transfère à un conseiller. Veuillez patienter...")
        return [SlotSet("rag_failure_count", 0)]

class ActionExplainSource(Action):
    def name(self) -> Text:
        return "action_explain_source"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        source = tracker.get_slot("last_source")
        msg = f"Source : {source}" if source else "Aucune source récente consultée."
        dispatcher.utter_message(text=msg)
        return []

class ActionViewTicket(Action):
    def name(self) -> Text:
        return "action_view_ticket"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        url = tracker.get_slot("glpi_ticket_url")
        if url:
            dispatcher.utter_message(text=f"Vous pouvez suivre votre demande ici : {url}")
        else:
            dispatcher.utter_message(text="Aucun ticket n'est ouvert pour le moment.")
        return []