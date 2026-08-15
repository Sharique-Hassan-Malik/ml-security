"""
Tests for the model extraction attack simulator.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from modelext.attack_loop import ExtractionAttack, ExtractionConfig
from modelext.trainer import SubstituteTrainer
from modelext.architectures import SubstituteCNN, SubstituteMLP, VictimCNN
from modelext.oracle import BlackBoxOracle, QueryBudgetExceeded
from modelext.strategies import AdaptiveStrategy, JacobianStrategy, RandomStrategy

INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 10


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def victim():
    torch.manual_seed(0)
    m = VictimCNN(num_classes=NUM_CLASSES)
    m.eval()
    return m


@pytest.fixture()
def soft_oracle(victim):
    return BlackBoxOracle(victim, hard_label=False)


@pytest.fixture()
def hard_oracle(victim):
    return BlackBoxOracle(victim, hard_label=True)


@pytest.fixture()
def substitute():
    torch.manual_seed(1)
    return SubstituteCNN(num_classes=NUM_CLASSES)


# ------------------------------------------------------------------
# Oracle
# ------------------------------------------------------------------

class TestOracle:
    def test_soft_label_shape(self, soft_oracle):
        x   = torch.rand(8, *INPUT_SHAPE)
        out = soft_oracle.query(x)
        assert out.shape == (8, NUM_CLASSES)

    def test_soft_label_sums_to_one(self, soft_oracle):
        x   = torch.rand(4, *INPUT_SHAPE)
        out = soft_oracle.query(x)
        assert torch.allclose(out.sum(dim=1), torch.ones(4), atol=1e-4)

    def test_hard_label_shape(self, hard_oracle):
        x   = torch.rand(6, *INPUT_SHAPE)
        out = hard_oracle.query(x)
        assert out.shape == (6,)
        assert out.dtype == torch.int64

    def test_hard_label_range(self, hard_oracle):
        x   = torch.rand(10, *INPUT_SHAPE)
        out = hard_oracle.query(x)
        assert out.min().item() >= 0
        assert out.max().item() < NUM_CLASSES

    def test_query_counter_increments(self, soft_oracle):
        x = torch.rand(5, *INPUT_SHAPE)
        soft_oracle.query(x)
        assert soft_oracle.query_count == 5
        soft_oracle.query(x)
        assert soft_oracle.query_count == 10

    def test_budget_exceeded(self, victim):
        oracle = BlackBoxOracle(victim, query_limit=3)
        x = torch.rand(2, *INPUT_SHAPE)
        oracle.query(x)
        with pytest.raises(QueryBudgetExceeded):
            oracle.query(x)

    def test_budget_remaining(self, victim):
        oracle = BlackBoxOracle(victim, query_limit=10)
        oracle.query(torch.rand(4, *INPUT_SHAPE))
        assert oracle.budget_remaining == 6

    def test_reset_counter(self, soft_oracle):
        soft_oracle.query(torch.rand(5, *INPUT_SHAPE))
        soft_oracle.reset_counter()
        assert soft_oracle.query_count == 0

    def test_temperature_increases_entropy(self, victim):
        x  = torch.rand(4, *INPUT_SHAPE)
        o1 = BlackBoxOracle(victim, temperature=1.0)
        o2 = BlackBoxOracle(victim, temperature=5.0)
        p1 = o1.query(x)
        p2 = o2.query(x)
        e1 = -(p1.clamp(1e-9) * p1.clamp(1e-9).log()).sum(dim=1).mean()
        e2 = -(p2.clamp(1e-9) * p2.clamp(1e-9).log()).sum(dim=1).mean()
        assert e2 > e1

    def test_num_classes(self, soft_oracle):
        assert soft_oracle.num_classes() == NUM_CLASSES


# ------------------------------------------------------------------
# Strategies
# ------------------------------------------------------------------

class TestRandomStrategy:
    def test_output_shape(self):
        s   = RandomStrategy(INPUT_SHAPE, batch_size=16)
        out = s.generate()
        assert out.shape == (16, *INPUT_SHAPE)

    def test_output_range(self):
        s   = RandomStrategy(INPUT_SHAPE, batch_size=8)
        out = s.generate()
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_no_substitute_needed(self):
        s   = RandomStrategy(INPUT_SHAPE, batch_size=4)
        out = s.generate(substitute=None, seed_data=None)
        assert out.shape[0] == 4


class TestJacobianStrategy:
    def test_shape_with_model(self, substitute):
        s    = JacobianStrategy(INPUT_SHAPE, batch_size=8)
        seed = torch.rand(16, *INPUT_SHAPE)
        out  = s.generate(substitute=substitute, seed_data=seed)
        assert out.shape == (8, *INPUT_SHAPE)

    def test_output_range(self, substitute):
        s    = JacobianStrategy(INPUT_SHAPE, batch_size=8)
        seed = torch.rand(16, *INPUT_SHAPE)
        out  = s.generate(substitute=substitute, seed_data=seed)
        assert out.min() >= 0.0 - 1e-5 and out.max() <= 1.0 + 1e-5

    def test_fallback_without_model(self):
        s   = JacobianStrategy(INPUT_SHAPE, batch_size=6)
        out = s.generate(substitute=None, seed_data=None)
        assert out.shape == (6, *INPUT_SHAPE)

    def test_fallback_without_seed(self, substitute):
        s   = JacobianStrategy(INPUT_SHAPE, batch_size=6)
        out = s.generate(substitute=substitute, seed_data=None)
        assert out.shape == (6, *INPUT_SHAPE)


class TestAdaptiveStrategy:
    def test_shape(self, substitute):
        s   = AdaptiveStrategy(INPUT_SHAPE, batch_size=8, n_candidates=64)
        out = s.generate(substitute=substitute)
        assert out.shape == (8, *INPUT_SHAPE)

    def test_output_range(self, substitute):
        s   = AdaptiveStrategy(INPUT_SHAPE, batch_size=8, n_candidates=64)
        out = s.generate(substitute=substitute)
        assert out.min() >= 0.0 - 1e-5 and out.max() <= 1.0 + 1e-5

    def test_with_seed_data(self, substitute):
        s    = AdaptiveStrategy(INPUT_SHAPE, batch_size=8, n_candidates=64)
        seed = torch.rand(32, *INPUT_SHAPE)
        out  = s.generate(substitute=substitute, seed_data=seed)
        assert out.shape == (8, *INPUT_SHAPE)

    def test_fallback_without_model(self):
        s   = AdaptiveStrategy(INPUT_SHAPE, batch_size=6, n_candidates=32)
        out = s.generate(substitute=None)
        assert out.shape == (6, *INPUT_SHAPE)


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class TestSubstituteTrainer:
    def test_train_soft_returns_loss(self, substitute, soft_oracle):
        x = torch.rand(32, *INPUT_SHAPE)
        y = soft_oracle.query(x)
        t = SubstituteTrainer(substitute, epochs=2, soft_labels=True)
        m = t.train_round(x, y)
        assert "final_loss" in m
        assert m["final_loss"] >= 0.0

    def test_train_hard_returns_loss(self, substitute, hard_oracle):
        x = torch.rand(32, *INPUT_SHAPE)
        y = hard_oracle.query(x)
        t = SubstituteTrainer(substitute, epochs=2, soft_labels=False)
        m = t.train_round(x, y)
        assert m["final_loss"] >= 0.0

    def test_fidelity_agreement_in_range(self, substitute, soft_oracle):
        x = torch.rand(32, *INPUT_SHAPE)
        y = soft_oracle.query(x)
        t = SubstituteTrainer(substitute, epochs=3)
        t.train_round(x, y)
        f = t.evaluate_fidelity(soft_oracle.query, x)
        assert 0.0 <= f["agreement"] <= 1.0

    def test_fidelity_kl_non_negative(self, substitute, soft_oracle):
        x = torch.rand(32, *INPUT_SHAPE)
        y = soft_oracle.query(x)
        t = SubstituteTrainer(substitute, epochs=3, soft_labels=True)
        t.train_round(x, y)
        f = t.evaluate_fidelity(soft_oracle.query, x)
        assert f["kl_divergence"] is not None
        assert f["kl_divergence"] >= 0.0

    def test_accumulation_uses_all_data(self, substitute, soft_oracle):
        x1 = torch.rand(16, *INPUT_SHAPE)
        y1 = soft_oracle.query(x1)
        x2 = torch.rand(16, *INPUT_SHAPE)
        y2 = soft_oracle.query(x2)
        import torch as T
        x_all = T.cat([x1, x2])
        y_all = T.cat([y1, y2])
        t = SubstituteTrainer(substitute, epochs=2)
        m = t.train_round(x2, y2, x_all=x_all, y_all=y_all)
        assert m["final_loss"] >= 0.0


# ------------------------------------------------------------------
# Full extraction loop
# ------------------------------------------------------------------

class TestExtractionLoop:
    def _quick_attack(self, victim, strategy_cls, **strategy_kwargs):
        oracle  = BlackBoxOracle(victim, hard_label=False)
        sub     = SubstituteCNN(num_classes=NUM_CLASSES)
        strat   = strategy_cls(INPUT_SHAPE, batch_size=16, **strategy_kwargs)
        trainer = SubstituteTrainer(sub, epochs=2)
        cfg     = ExtractionConfig(
            n_rounds=3, queries_per_round=16, eval_size=20, seed=0
        )
        attack = ExtractionAttack(oracle, sub, strat, trainer, cfg)
        return attack.run(verbose=False), attack

    def test_random_runs(self, victim):
        result, _ = self._quick_attack(victim, RandomStrategy)
        assert len(result.rounds) == 3
        assert result.total_queries > 0
        assert 0.0 <= result.final_agreement <= 1.0

    def test_jacobian_runs(self, victim):
        result, _ = self._quick_attack(victim, JacobianStrategy)
        assert len(result.rounds) == 3

    def test_adaptive_runs(self, victim):
        result, _ = self._quick_attack(
            victim, AdaptiveStrategy, n_candidates=64
        )
        assert len(result.rounds) == 3

    def test_queries_monotone(self, victim):
        result, _ = self._quick_attack(victim, RandomStrategy)
        qs = [r.queries_used for r in result.rounds]
        assert qs == sorted(qs)

    def test_to_dict_structure(self, victim):
        result, attack = self._quick_attack(victim, RandomStrategy)
        d = attack.to_dict(result)
        assert "total_queries"   in d
        assert "final_agreement" in d
        assert "rounds"          in d
        assert len(d["rounds"])  == 3
        for r in d["rounds"]:
            assert "agreement"    in r
            assert "queries_used" in r
            assert "train_loss"   in r

    def test_budget_stops_early(self, victim):
        oracle  = BlackBoxOracle(victim, hard_label=False, query_limit=40)
        sub     = SubstituteCNN(num_classes=NUM_CLASSES)
        strat   = RandomStrategy(INPUT_SHAPE, batch_size=16)
        trainer = SubstituteTrainer(sub, epochs=1)
        cfg     = ExtractionConfig(
            n_rounds=10, queries_per_round=16, eval_size=20, seed=0
        )
        attack = ExtractionAttack(oracle, sub, strat, trainer, cfg)
        result = attack.run(verbose=False)
        assert len(result.rounds) < 10

    def test_hard_label_mode(self, victim):
        oracle  = BlackBoxOracle(victim, hard_label=True)
        sub     = SubstituteCNN(num_classes=NUM_CLASSES)
        strat   = RandomStrategy(INPUT_SHAPE, batch_size=16)
        trainer = SubstituteTrainer(sub, epochs=2, soft_labels=False)
        cfg     = ExtractionConfig(
            n_rounds=2, queries_per_round=16, eval_size=20, seed=1
        )
        attack = ExtractionAttack(oracle, sub, strat, trainer, cfg)
        result = attack.run(verbose=False)
        assert len(result.rounds) == 2

    def test_mlp_substitute(self, victim):
        oracle  = BlackBoxOracle(victim)
        sub     = SubstituteMLP(num_classes=NUM_CLASSES)
        strat   = RandomStrategy(INPUT_SHAPE, batch_size=16)
        trainer = SubstituteTrainer(sub, epochs=2)
        cfg     = ExtractionConfig(
            n_rounds=2, queries_per_round=16, eval_size=20, seed=2
        )
        attack = ExtractionAttack(oracle, sub, strat, trainer, cfg)
        result = attack.run(verbose=False)
        assert 0.0 <= result.final_agreement <= 1.0
