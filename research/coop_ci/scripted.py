"""Scripted cooks: persona twins (partner) and allocation policies for the ego (Gate 0).

Every scripted cook shares one competence skeleton (deliver dishes, plate with a plate in
hand, load the held ingredient into a pot whose open order still needs it, reuse items set
aside on counters) and differs only in the priority of its jobs when its hands are empty:

  plate_ready     a soup is ready and no free plate is out        -> fetch a plate
  plate_cooking   a soup is cooking or ready and no free plate     -> fetch a plate
  fetch:<ing>     the open pots still need <ing>                   -> fetch it
  fetch_missing   the open pots still need any ingredient          -> fetch the most needed one

"Still need" counts what the pots hold and what the teammate is carrying (personas and
partner-aware egos), or only the pots (`ignore_partner`, the neutral ego). Speed and
competence are identical across personas; only the division of labour differs.

Ego allocation policies for persona X (Gate 0):
  complementary(X) = the jobs of X's complement persona (personas.yaml `complement`)
  conflicting(X)   = X's own jobs (copies the partner's tendency)
  neutral          = greedy on pot needs, ignores the partner
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from .env import CoopEnv, decode_ingredients, is_cooked, is_ingredient, is_plate
from .planner import acquires, reachable_targets, side_cell
from .policy import Decision, Policy

NEUTRAL_JOBS = ["plate_ready", "fetch_missing", "plate_cooking"]


def _other(env: CoopEnv, i: int) -> int:
    return 1 - i


def _plate_demand(env: CoopEnv, cooking_too: bool) -> int:
    n = 0
    for p in env.pots:
        st = env.pot_status(p)[0]
        n += st == "ready" or (cooking_too and st == "cooking")
    return n


def _plates_out(env: CoopEnv, i: int, ignore_partner: bool) -> int:
    """Free (empty) plates already out: on counters, plus the partner's unless ignored."""
    n = sum(1 for o in env.counter_items().values() if is_plate(o) and not is_cooked(o))
    if not ignore_partner:
        inv = env.inventory(_other(env, i))
        n += is_plate(inv) and not is_cooked(inv)
    return n


def is_junk(pp: dict) -> bool:
    """An idle pot with ingredients that fit no open order."""
    return pp["status"] == "idle" and bool(pp["contents"]) and pp["target"] is None


def needs(env: CoopEnv, i: int, ignore_partner: bool = False) -> Counter:
    return env.missing_ingredients(() if ignore_partner else (_other(env, i),))


def from_counter(env: CoopEnv, i: int, kind: str) -> Optional[str]:
    """'pick up from counter x y' for a reachable counter holding this item kind, if any."""
    def k(obj):
        if is_cooked(obj):
            return "dish"
        if is_plate(obj):
            return "plate"
        ings = decode_ingredients(obj)
        return env.ing_name(ings[0]) if len(ings) == 1 else ""
    held = [c for c, o in env.counter_items().items() if env.in_view(i, c) and k(o) == kind]
    reach = reachable_targets(env, i, held)
    return f"pick up from counter {reach[0][0]} {reach[0][1]}" if reach else None


def job_action(env: CoopEnv, i: int, job: str, ignore_partner: bool = False) -> Optional[str]:
    """The macro that serves `job` now (hands empty), or None if the job does not apply."""
    if job in ("plate_ready", "plate_cooking"):
        if _plate_demand(env, job == "plate_cooking") > _plates_out(env, i, ignore_partner):
            return from_counter(env, i, "plate") or "fetch plate"
        return None
    need = needs(env, i, ignore_partner)
    if job.startswith("fetch:"):
        name = job.split(":", 1)[1]
        k = env.ingredient_names.index(name)
        if need[k] > 0 and k in env.piles:
            return from_counter(env, i, name) or f"fetch {name}"
        return None
    if job == "fetch_missing":
        cand = sorted((k for k in need if need[k] > 0 and k in env.piles), key=lambda k: (-need[k], k))
        if cand:
            name = env.ing_name(cand[0])
            return from_counter(env, i, name) or f"fetch {name}"
        return None
    raise ValueError(f"unknown job {job}")


class ScriptedCook(Policy):
    role = "scripted"

    def __init__(self, jobs: list[str], ignore_partner: bool = False, name: str = "scripted"):
        self.jobs, self.ignore_partner, self.name = list(jobs), ignore_partner, name

    def decide(self, env, i, last_result="", context=None) -> Decision:
        return Decision(env.t, i, self.act(env, i))

    @staticmethod
    def _free_counter(env: CoopEnv, i: int) -> bool:
        return bool(reachable_targets(env, i, [c for c in env.counters if env.cell(c)[1] == 0]))

    def act(self, env: CoopEnv, i: int) -> str:
        inv = env.inventory(i)
        many = len(env.pots) > 1
        if is_cooked(inv):
            dish = decode_ingredients(inv)
            if env.matches_open_order(dish) or not self._free_counter(env, i):
                return "deliver"  # a wrong dish with nowhere to set it aside is served (-20)
            return "place on counter"
        if is_plate(inv):
            st = {p: env.pot_status(p) for p in env.pots}
            ready = [p for p in env.pots if st[p][0] == "ready"]
            cooking = sorted((p for p in env.pots if st[p][0] == "cooking"), key=lambda p: st[p][2])
            pots = ready or cooking
            if not pots:
                return "wait 2"
            return f"plate soup {env.pot_name[pots[0]]}" if many else "plate soup"
        if is_ingredient(inv):
            k = decode_ingredients(inv)[0]
            pots = env.pot_plan()
            plan = [pp for pp in pots if pp["status"] == "idle" and k in pp["missing"]]
            # A pot whose contents fit no open order (e.g. both cooks added the last unit at
            # once) is flushed: fill it with anything, then plate the wrong soup and set it aside.
            plan = plan or [pp for pp in pots if is_junk(pp)]
            if not plan:
                return "place on counter" if self._free_counter(env, i) else "put back"
            order = {p: n for n, p in enumerate(reachable_targets(env, i, [pp["pot"] for pp in plan]))}
            plan.sort(key=lambda pp: (-len(pp["contents"]), order.get(pp["pot"], 99)))
            pot = plan[0]["pot"]
            # Walk next to the pot first and decide while looking at it.
            side = side_cell(env, i, pot)
            if side is not None and env.agent_pos(i) != side:
                return f"move to {side[0]} {side[1]}"
            return f"put in {env.pot_name[pot]}" if many else "put in pot"
        if inv != 0:
            return "place on counter"
        if any(is_junk(pp) for pp in env.pot_plan()):
            name = env.ing_name(min(env.piles))
            return from_counter(env, i, name) or f"fetch {name}"
        for job in self.jobs:
            a = job_action(env, i, job, self.ignore_partner)
            if a:
                return a
        return "wait 2"


def persona_policy(personas: dict, pid: str) -> ScriptedCook:
    sc = personas[pid]["scripted"]
    return ScriptedCook(sc["jobs"], ignore_partner=sc.get("ignore_partner", False), name=pid)


def ego_policy(personas: dict, kind: str, persona: Optional[str] = None) -> ScriptedCook:
    """complementary / conflicting (to `persona`), neutral, or `as:<persona>` (that persona's jobs)."""
    if kind == "neutral":
        return ScriptedCook(NEUTRAL_JOBS, ignore_partner=True, name="neutral")
    if kind == "complementary":
        comp = personas[persona]["complement"]
        p = persona_policy(personas, comp)
        p.name = f"complementary({persona})"
        return p
    if kind == "conflicting":
        p = persona_policy(personas, persona)
        p.name = f"conflicting({persona})"
        return p
    if kind.startswith("as:"):
        return persona_policy(personas, kind[3:])
    raise ValueError(f"unknown scripted ego {kind}")


# --------------------------------------------------------------------------- #
# Opportunity predicates (persona compliance) and useful actions (dependency delay)
# --------------------------------------------------------------------------- #
def opportunity(env: CoopEnv, i: int, persona: dict) -> Optional[set]:
    """At a decision of cook i with persona `persona`: if its tendency applies now, the set of
    items whose acquisition counts as following it; else None."""
    if env.inventory(i) != 0:
        return None
    kind = persona.get("opportunity")
    plates_short = lambda cooking: _plate_demand(env, cooking) > _plates_out(env, i, False)  # noqa: E731
    if kind == "needs_preferred":
        pref = persona["scripted"]["preferred"]
        k = env.ingredient_names.index(pref)
        if needs(env, i)[k] > 0 and not plates_short(False):
            return {pref}
        return None
    if kind == "pots_need_ingredients":
        need = needs(env, i)
        return {env.ing_name(k) for k in need} if sum(need.values()) else None
    if kind == "soup_needs_plate":
        return {"plate"} if plates_short(True) else None
    return None


def complies(env: CoopEnv, macro, allowed: set) -> bool:
    return acquires(env, macro) in allowed


def useful_actions(env: CoopEnv, i: int, partner_subgoal: Optional[str] = None) -> set:
    """Coarse subgoals that would advance an open order now without duplicating what the
    partner is carrying or committed to (used for 'idle while useful')."""
    inv = env.inventory(i)
    out = set()
    if is_cooked(inv):
        if env.matches_open_order(decode_ingredients(inv)):
            out.add("deliver")
        return out
    if is_plate(inv):
        if _plate_demand(env, True):
            out.add("plate")
        return out
    if is_ingredient(inv):
        k = decode_ingredients(inv)[0]
        if any(pp["status"] == "idle" and k in pp["missing"] for pp in env.pot_plan()):
            out.add("load")
        return out
    j = _other(env, i)
    need = needs(env, i)
    plates = _plates_out(env, i, False)
    if partner_subgoal and env.inventory(j) == 0:
        if partner_subgoal == "fetch:plate":
            plates += 1
        elif partner_subgoal.startswith("fetch:"):
            name = partner_subgoal.split(":", 1)[1]
            if name in env.ingredient_names:
                need[env.ingredient_names.index(name)] -= 1
    out |= {f"fetch:{env.ing_name(k)}" for k, v in need.items() if v > 0}
    if _plate_demand(env, True) > plates:
        out.add("fetch:plate")
    return out
