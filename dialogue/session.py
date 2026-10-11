import re
import random

from llm.prompts import CROP_DETECTION_PROMPT, DISAMBIGUATION_PROMPT, FOLLOWUP_QA_PROMPT, INTENT_CLASSIFICATION_PROMPT, Summarize_Symptoms_PROMPT, QUESTION_RESOLUTION_PROMPT
from llm.client import get_fast_llm_completion, get_reasoning_completion
from rag.retriever import get_all_candidates, get_disease_record

# ---------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------

MAX_DISAMBIGUATION_QUESTIONS = 4
MAX_FOLLOWUPS = 4

GENERIC_ERROR_REPLY = "Maaf kijiyega, thodi dikkat aa gayi. Kripya dobara boliye."
NO_MATCH_REPLY = "Maaf kijiyega, lakshan ke aadhar par main sahi bimari nahi pehchaan paa raha hoon. Kripya krishi visheshagya se sampark karein."
CROP_UNCLEAR_REPLY = "Maaf kijiyega, main theek se samajh nahi paaya. Kya aap bata sakte hain ki yeh samasya dhaan ki hai ya kapas ki?"
FOLLOWUP_ELICIT_REPLY = "Aap kya jaanna chahenge — lakshan, karan, ya ilaj ke baare mein?"

CROP_HINDI_LABEL = {
    "RICE": "dhaan",
    "COTTON": "kapas",
}

BARE_AFFIRMATION_WORDS = {
    "yes", "yeah", "yep", "sure", "ok", "okay", "please", "alright", "fine",
    "haan", "ha", "haa", "ji", "bilkul", "theek", "thik", "sahi",
    "हाँ", "जी", "ठीक", "बिल्कुल", "सही"
}


class DialogueManager:
    def __init__(self):
        self.phase = "crop_detection"
        self.crop = None
        self.symptoms_text = ""
        self.qa_history = []
        self.questions_asked = 0
        self.identified_disease = None
        self.followups_asked = 0
        self.candidates_text = ""
        self.candidate_names = []

        # The question we most recently asked and are waiting on a reply to.
        # Cleared once resolved (YES/NO) or once we move past symptom_diagnosis.
        self.pending_question = None

        # Every discriminator the farmer was NOT able to answer, ever, this
        # call. These are permanently off-limits for re-asking (reworded or
        # otherwise) — see DISAMBIGUATION_PROMPT's PENDING DISCRIMINATOR CHECK.
        self.unresolved_features = []

        self._state_handlers = {
            "crop_detection": self._handle_crop_detection,
            "symptom_diagnosis": self._handle_symptom_diagnosis,
            "followup": self._handle_followup,
            "followup_elicit": self._handle_followup_elicit,
            "completed": self._handle_completed,
        }

    async def process_turn(self, farmer_text_raw: str) -> str:
        """Main entry point. Routes directly to the current state handler."""
        handler = self._state_handlers.get(self.phase)

        if not handler:
            return GENERIC_ERROR_REPLY

        try:
            return await handler(farmer_text_raw)
        except Exception as e:
            print(
                f"[DialogueManager] Unhandled error in phase={self.phase}: {e}")
            return GENERIC_ERROR_REPLY

    # ---------------------------------------------------------
    # STATE HANDLERS
    # ---------------------------------------------------------

    async def _handle_completed(self, text: str) -> str:
        return "Yeh call samapt ho chuki hai. Naye sawaal ke liye kripya naya call karein."

    async def _handle_crop_detection(self, text: str) -> str:
        """Phase 0: Figure out the crop from raw text."""
        self.symptoms_text += f" {text}".strip()

        prompt = CROP_DETECTION_PROMPT.format(farmer_text=self.symptoms_text)
        llm_output = await get_fast_llm_completion(prompt, temperature=0.0)
        decision = llm_output.strip().upper()

        if "OTHER_CROP" in decision:
            self.phase = "completed"
            return "Maaf kijiyega, humari helpline abhi sirf Dhaan aur Kapas ki bimariyon ke baare mein jaankari de sakti hai. Call karne ke liye dhanyawad, namaste."

        elif "RICE" in decision or "COTTON" in decision:
            self.crop = "RICE" if "RICE" in decision else "COTTON"
            return await self._initiate_diagnosis()

        return CROP_UNCLEAR_REPLY

    async def _initiate_diagnosis(self) -> str:
        """Helper to transition from Crop Detection to Diagnosis."""
        self.phase = "symptom_diagnosis"
        crop_query = self.crop.lower()
        candidate_records = get_all_candidates(crop_query)

        if not candidate_records:
            self.phase = "completed"
            return "Maaf kijiyega, abhi mere paas is fasal ki bimariyon ki jaankari uplabdh nahi hai. Kripya krishi visheshagya se sampark karein."

        # Shuffle so the model can't develop a first-listed-wins bias
        # (e.g. Blast always being candidate #1 for rice).
        candidate_records = list(candidate_records)
        random.shuffle(candidate_records)

        self.candidate_names = [r.get("disease_name", "")
                                for r in candidate_records]
        self.candidates_text = self._format_candidates_text(candidate_records)

        response = await self._trigger_llm_turn()

        label = CROP_HINDI_LABEL.get(self.crop, self.crop)
        return f"Achha, {label} ki fasal hai. {response}"

    async def _handle_symptom_diagnosis(self, text: str) -> str:
        """Phase 1: Handles the farmer's answers to our clarifying questions."""
        self.qa_history.append(f"Farmer: {text}")

        # Before calling the (expensive) reasoning model, resolve whether the
        # farmer's reply actually answered the question we just asked. This
        # is a small, deterministic check — same pattern as the bare
        # affirmation / intent-classification gates below — so the big
        # disambiguation prompt is handed a hard fact instead of having to
        # infer it from free-text history.
        pending_status = "NONE"
        if self.pending_question:
            resolution_prompt = QUESTION_RESOLUTION_PROMPT.format(
                pending_question=self.pending_question,
                farmer_reply=text,
            )
            raw = await get_fast_llm_completion(resolution_prompt, temperature=0.0)
            status = raw.strip().upper()
            pending_status = status if status in (
                "YES", "NO", "UNRESOLVED") else "UNRESOLVED"

            if pending_status == "UNRESOLVED":
                # Retire this discriminator permanently — the farmer
                # couldn't answer it once, so re-asking it (even reworded)
                # is not useful and reads as not listening.
                self.unresolved_features.append(self.pending_question)

        return await self._trigger_llm_turn(pending_status=pending_status)

    async def _handle_followup(self, text: str) -> str:
        """Phase 3 gate: check if it's a bare 'yes' or a real question."""
        if self._is_bare_affirmation(text):
            self.phase = "followup_elicit"
            return FOLLOWUP_ELICIT_REPLY

        intent_prompt = INTENT_CLASSIFICATION_PROMPT.format(farmer_reply=text)
        intent_response = await get_fast_llm_completion(intent_prompt, temperature=0.0)

        if "NO" in intent_response.strip().upper():
            self.phase = "completed"
            return "Theek hai. KrishiSeva mein call karne ke liye dhanyawad. Namaste!"

        return await self._answer_followup_question(text)

    async def _handle_followup_elicit(self, text: str) -> str:
        """Phase 3b: Handles the user's actual question after a bare 'yes'."""
        return await self._answer_followup_question(text)

    # ---------------------------------------------------------
    # CORE LOGIC HELPERS
    # ---------------------------------------------------------

    async def _trigger_llm_turn(self, pending_status: str = "NONE") -> str:
        """The Reasoning Engine for Phase 1 (Disambiguation)."""
        formatted_history = "\n".join(self.qa_history)
        unresolved_features_text = (
            "\n".join(f"- {q}" for q in self.unresolved_features)
            if self.unresolved_features else "(none)"
        )

        prompt = DISAMBIGUATION_PROMPT.format(
            symptom_text=self.symptoms_text,
            candidates=self.candidates_text,
            qa_history=formatted_history,
            questions_asked_so_far=self.questions_asked,
            max_questions=MAX_DISAMBIGUATION_QUESTIONS,
            pending_question=self.pending_question or "(none)",
            pending_question_status=pending_status,
            unresolved_features=unresolved_features_text,
        )

        llm_output = await get_reasoning_completion(prompt)
        analysis, action, content = self._parse_analysis_action_content(
            llm_output)

        # Best debugging signal for "why did it pick this" — log it, never
        # surface it to the farmer.
        # try:
        #     print(f"[debug] analysis: {analysis}")
        # except UnicodeEncodeError:
        #     import sys
        #     sys.stdout.buffer.write((f"[debug] analysis: {analysis}\n").encode("utf-8", errors="replace"))
        #     sys.stdout.buffer.flush()

        if not action:
            return "Maaf kijiye, main theek se samajh nahi paaya. Kya aap lakshan dobara bata sakte hain?"

        if action == "ASK":
            if self.questions_asked >= MAX_DISAMBIGUATION_QUESTIONS:
                self.phase = "completed"
                return NO_MATCH_REPLY

            self.qa_history.append(f"Agent: {content}")
            self.questions_asked += 1
            self.pending_question = content
            return content

        elif action == "ANSWER":
            self.pending_question = None
            if content not in self.candidate_names:
                self.phase = "completed"
                return NO_MATCH_REPLY
            self.identified_disease = content
            return await self._announce_diagnosis()

        elif action == "NO_MATCH":
            self.pending_question = None
            self.phase = "completed"
            return NO_MATCH_REPLY

        return "Maaf kijiye, main theek se samajh nahi paaya. Kya aap lakshan dobara bata sakte hain?"

    async def _announce_diagnosis(self) -> str:
        """Phase 2: Announce the matched disease and its brief symptoms."""
        kb_record = get_disease_record(self.identified_disease)

        if not kb_record:
            self.phase = "completed"
            return f"Lagta hai yeh {self.identified_disease} hai, par mere paas iski poori dawai ki jankari abhi nahi hai. Kripya kisi krishi visheshagya se sampark karein."

        self.identified_disease = kb_record.get(
            "disease_name", self.identified_disease)

        bullets = kb_record.get("symptoms_bullets", [])
        symptoms_text = " ".join(
            bullets) if bullets else "Lakshan ki jankari uplabdh nahi hai."

        summary_prompt = Summarize_Symptoms_PROMPT.format(
            disease=self.identified_disease,
            symptoms_text=symptoms_text
        )
        symptom_summary = await get_fast_llm_completion(summary_prompt, temperature=0.1)

        self.phase = "followup"

        return f"Aapke bataye gaye lakshano ke aadhar par, yeh samasya {self.identified_disease} lagti hai. {symptom_summary}\n\nKya aapko {self.identified_disease} ke baare mein jankari chahiye?"

    async def _answer_followup_question(self, text: str) -> str:
        """Handles answering the farmer's question about the disease."""
        self.followups_asked += 1
        if self.followups_asked > MAX_FOLLOWUPS:
            self.phase = "completed"
            return "Maaf kijiyega, is call ka samay samapt ho gaya hai. Aur jaankari ke liye kripya naya call karein. Dhanyawad!"

        kb_record = get_disease_record(self.identified_disease)
        qa_prompt = FOLLOWUP_QA_PROMPT.format(
            disease=self.identified_disease,
            kb_context=self._format_kb_record(kb_record),
            farmer_question=text
        )

        answer_text = await get_fast_llm_completion(qa_prompt, temperature=0.1)
        self.phase = "followup"
        return f"{answer_text}\n\nKya apko iske baare mein aur kuch janna hai?"

    # ---------------------------------------------------------
    # UTILITY HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _is_bare_affirmation(text: str) -> bool:
        normalized = re.sub(r"[^\w\s]", "", text.strip().lower())
        words = normalized.split()
        if not words:
            return False
        return all(w in BARE_AFFIRMATION_WORDS for w in words)

    @staticmethod
    def _format_kb_record(kb_record: dict) -> str:
        if not kb_record:
            return "No knowledge base record available."

        bullets = kb_record.get("symptoms_bullets", [])
        symptoms = " ".join(bullets) if bullets else "Not available."

        return "\n".join([
            f"Disease: {kb_record.get('disease_name', 'Unknown')}",
            f"Causal organism: {kb_record.get('causal_organism', 'Not available.')}",
            f"Symptoms: {symptoms}",
            f"Favourable conditions: {kb_record.get('favourable_conditions') or 'Not available.'}",
            f"Survival and spread: {kb_record.get('survival_and_spread') or 'Not available.'}",
            f"Management: {kb_record.get('management') or 'Not available.'}",
        ])

    @staticmethod
    def _format_candidates_text(candidate_records: list[dict]) -> str:
        lines = []
        for i, record in enumerate(candidate_records, start=1):
            name = record.get("disease_name", "Unknown")
            bullets = record.get("symptoms_bullets", [])
            symptoms = " ".join(bullets) if bullets else "Not available."
            lines.append(f"{i}. {name}: {symptoms}")
        return "\n".join(lines)

    @staticmethod
    def _parse_analysis_action_content(llm_output: str):
        """Parses the required ANALYSIS / ACTION / CONTENT three-line output.

        ANALYSIS is logged for debugging only — never shown to the farmer.
        This replaces the old two-line ACTION/CONTENT parser: forcing the
        model to state its analysis as graded output (rather than invisible
        reasoning it could skip) is what actually made it check whether the
        pending question was resolved before answering.
        """
        match = re.search(
            r"ANALYSIS:\s*(.*?)\s*\n+ACTION:\s*(\w+)\s*\n+CONTENT:\s*(.*)",
            llm_output.strip(), re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None, None, None
        return match.group(1).strip(), match.group(2).strip().upper(), match.group(3).strip()
