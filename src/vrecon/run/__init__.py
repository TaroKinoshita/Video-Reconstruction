"""Run manager: proposals, constraints, accept rule, atomic adoption (PLAN.md 6)."""

from .manager import AcceptRule, Change, Outcome, Proposal, RunManager, apply_changes, changed_fields, check_constraints

__all__ = ["AcceptRule", "Change", "Outcome", "Proposal", "RunManager", "apply_changes", "changed_fields",
           "check_constraints"]
