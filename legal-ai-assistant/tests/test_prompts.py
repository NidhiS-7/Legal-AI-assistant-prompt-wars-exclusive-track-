from core import prompts


def test_system_prompt_forbids_fabrication_and_advice():
    system = prompts.SYSTEM_PROMPT.lower()
    assert "not a substitute" in system or "not legal advice" in system
    assert "fabricate" in system or "invent" in system


def test_disclaimer_present_and_non_empty():
    assert prompts.DISCLAIMER.strip() != ""
    assert "not legal advice" in prompts.DISCLAIMER.lower()


def test_build_simplify_prompt_includes_document_text():
    text = "Tenant shall pay rent on the first of each month."
    prompt = prompts.build_simplify_prompt(text)
    assert text in prompt
    assert "plain-language" in prompt or "plain language" in prompt.lower()


def test_build_risk_clause_prompt_includes_categories():
    prompt = prompts.build_risk_clause_prompt("Some clause text")
    assert "HIGH ATTENTION" in prompt
    assert "STANDARD OBLIGATIONS" in prompt
    assert "PROTECTIVE" in prompt


def test_build_compare_prompt_includes_both_documents_and_labels():
    prompt = prompts.build_compare_prompt(
        "Doc A text", "Doc B text", "Lease v1", "Lease v2"
    )
    assert "Doc A text" in prompt
    assert "Doc B text" in prompt
    assert "Lease v1" in prompt
    assert "Lease v2" in prompt


def test_build_qa_prompt_includes_question_and_context():
    prompt = prompts.build_qa_prompt("Rent is $1000/month.", "How much is rent?")
    assert "Rent is $1000/month." in prompt
    assert "How much is rent?" in prompt


def test_build_qa_prompt_includes_history_when_provided():
    prompt = prompts.build_qa_prompt(
        "context", "next question", chat_history="user: earlier question"
    )
    assert "earlier question" in prompt


def test_build_checklist_prompt_has_three_sections():
    prompt = prompts.build_checklist_prompt("Some contract text")
    assert "BEFORE SIGNING" in prompt
    assert "AFTER SIGNING" in prompt
    assert "RED FLAGS" in prompt


def test_build_lawyer_prep_prompt_includes_user_context_when_given():
    prompt = prompts.build_lawyer_prep_prompt("contract text", "I'm worried about the NDA clause")
    assert "I'm worried about the NDA clause" in prompt
    assert "contract text" in prompt


def test_build_lawyer_prep_prompt_without_context_still_valid():
    prompt = prompts.build_lawyer_prep_prompt("contract text")
    assert "contract text" in prompt
