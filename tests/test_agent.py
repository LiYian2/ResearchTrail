import pytest

from agent.planner import AgentPlanner


class TestAgentPlanner:
    def test_detect_intent_build_path(self):
        agent = AgentPlanner()
        intent = agent._detect_intent("I want to enter the field of Deep CFR")
        from shared.types import IntentType
        assert intent == IntentType.BUILD_READING_PATH

    def test_detect_intent_explain(self):
        agent = AgentPlanner()
        intent = agent._detect_intent("Explain the paper about CFR")
        from shared.types import IntentType
        assert intent == IntentType.EXPLAIN_PAPER

    def test_detect_intent_filter(self):
        agent = AgentPlanner()
        intent = agent._detect_intent("Show papers after 2020")
        from shared.types import IntentType
        assert intent == IntentType.FILTER_BY_YEAR

    def test_detect_intent_bridge(self):
        agent = AgentPlanner()
        intent = agent._detect_intent("Find bridge papers between CFR and deep RL")
        from shared.types import IntentType
        assert intent == IntentType.FIND_BRIDGE_PAPERS

    def test_detect_intent_visualize(self):
        agent = AgentPlanner()
        intent = agent._detect_intent("visualize the citation network")
        from shared.types import IntentType
        assert intent == IntentType.VISUALIZE

    def test_extract_profile_beginner(self):
        agent = AgentPlanner()
        profile = agent._extract_profile("I am new to imperfect-information games")
        from shared.types import UserLevel
        assert profile.user_level == UserLevel.BEGINNER

    def test_extract_profile_advanced(self):
        agent = AgentPlanner()
        profile = agent._extract_profile("State of the art in deep reinforcement learning for games")
        from shared.types import UserLevel
        assert profile.user_level == UserLevel.ADVANCED

    def test_clean_topic(self):
        agent = AgentPlanner()
        topic = agent._clean_topic("I want to enter the field of Deep CFR and its applications")
        assert "enter" not in topic.lower()
        assert "Deep CFR" in topic

    def test_clean_topic_preserves_core(self):
        agent = AgentPlanner()
        topic = agent._clean_topic("Build me a reading path for graph anomaly detection")
        assert "graph anomaly detection" in topic.lower()

    def test_clean_topic_deeply_understand(self):
        agent = AgentPlanner()
        topic = agent._clean_topic("I am a beginner and want to deeply understand Vision Transformer")
        assert topic == "Vision Transformer"

    def test_cfr_adaptive_expansion_is_domain_specific(self):
        agent = AgentPlanner()
        queries = agent._adaptive_expansion_queries("Counterfactual regret minimization")
        joined = " ".join(queries).lower()
        assert "extensive-form" in joined
        assert "poker" in joined
        assert "related work" not in joined
