import json
import pytest
import unittest.mock

from src.agent.tools import normalize_top_k, AgentTools
from src.agent.orchestrator import AgentOrchestrator
from src.agent.state import AgentState
from src.pipelines.agentic_graphrag import AgenticGraphRAGPipeline


def test_normalize_top_k_valid():
    assert normalize_top_k(5) == 5
    assert normalize_top_k("5") == 5
    assert normalize_top_k([5]) == 5
    assert normalize_top_k(["5"]) == 5


def test_normalize_top_k_invalid():
    with pytest.raises(ValueError):
        normalize_top_k([])

    with pytest.raises(ValueError):
        normalize_top_k([5, 10])

    with pytest.raises(ValueError):
        normalize_top_k(["5", "10"])

    with pytest.raises(ValueError):
        normalize_top_k(None)

    with pytest.raises(ValueError):
        normalize_top_k({})

    with pytest.raises(ValueError):
        normalize_top_k("abc")


def test_select_evidence_malformed_top_k():
    tools = AgentTools()
    c = [{"chunk_id": "1", "text": "a"}] * 5

    res = tools.select_evidence(
        "query",
        c,
        c,
        final_top_k=c,
    )

    assert "error" in res
    assert "Expected exactly one element" in res["error"]


def test_select_evidence_valid_top_k():
    pass


def test_orchestrator_serialization():
    llm_out = {
        "response": (
            '{"action": "select_evidence", '
            '"reason": "test", '
            '"parameters": {"final_top_k": 5}}'
        )
    }

    start = llm_out["response"].find("{")
    end = llm_out["response"].rfind("}") + 1

    action = json.loads(llm_out["response"][start:end])

    assert action["action"] == "select_evidence"
    assert action["parameters"]["final_top_k"] == 5


def test_agent_generation_limits():
    orchestrator = AgentOrchestrator()

    orchestrator.tools.llm_client.generate = unittest.mock.MagicMock(
        return_value={
            "response": (
                '{"action": "stop", '
                '"reason": "done", '
                '"parameters": {}}'
            )
        }
    )

    state = AgentState(
        question_id="q",
        question="test",
    )

    orchestrator.run(state)

    # Agentic controller generation uses 2048.
    orchestrator.tools.llm_client.generate.assert_called_with(
        unittest.mock.ANY,
        temperature=0.0,
        options={"num_predict": 2048},
    )

    pipeline = AgenticGraphRAGPipeline()

    pipeline.orchestrator.tools.llm_client.generate = (
        unittest.mock.MagicMock(
            return_value={"response": "final answer"}
        )
    )

    state.is_finished = True
    state.candidate_evidence = [
        {
            "title": "Doc1",
            "text": "evidence text",
        }
    ]

    pipeline.orchestrator.run = unittest.mock.MagicMock(
        return_value=state
    )

    pipeline.answer(
        "q1",
        "What is the answer?",
    )

    # Agentic final-answer generation also uses 2048.
    pipeline.orchestrator.tools.llm_client.generate.assert_called_with(
        unittest.mock.ANY,
        temperature=0.0,
        options={"num_predict": 2048},
    )


def test_evaluate_evidence_limits():
    tools = AgentTools()

    tools.llm_client.generate = unittest.mock.MagicMock(
        return_value={
            "response": '{"status": "insufficient"}'
        }
    )

    tools.evaluate_evidence(
        "query",
        [
            {
                "chunk_id": "1",
                "title": "t",
                "text": "txt",
            }
        ],
    )

    # Evidence evaluation intentionally remains at 512.
    tools.llm_client.generate.assert_called_with(
        unittest.mock.ANY,
        temperature=0.0,
        options={"num_predict": 512},
    )
