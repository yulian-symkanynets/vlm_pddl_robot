"""Interchangeable planning strategies behind one interface.

Every strategy takes the same semantic Scene and returns a Proposal (a list of
Move actions). `run()` times the strategy and checks the proposal with the one
shared validator, so results from different strategies are directly comparable
and nothing unvalidated reaches the robot controller.
"""
from .base import Outcome, Proposal, Strategy, run  # noqa: F401
