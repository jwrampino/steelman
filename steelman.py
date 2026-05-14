import os
import json
import uuid
import re
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# === Config =================================================

load_dotenv()

APP_TITLE   = "SteelMan"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BANK_PATH   = "steelman_bank.json"
ARG_STORE_PATH = "steelman_args.json"
POLICY_PATH = "steelman_policy.pt"

EMBED_DIM  = 384
TOP_K      = 8
HIDDEN_DIM = 64
LR         = 1e-3
GAMMA      = 1.0
EPS        = 1e-8

HARM_THRESHOLD = 0.8

# I only return an exemplar if it clears this cosine similarity floor.
EXEMPLAR_THRESHOLD = 0.75
MAX_EXEMPLARS      = 2

OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

REVISION_MODEL_ANTHROPIC = "claude-haiku-4-5-20251001"
MUTATE_MODEL_ANTHROPIC   = "claude-haiku-4-5-20251001"
CHECK_MODEL_ANTHROPIC    = "claude-haiku-4-5-20251001"
REVISION_MODEL_OPENAI    = "gpt-4o-mini"
MUTATE_MODEL_OPENAI      = "gpt-4o-mini"
CHECK_MODEL_OPENAI       = "gpt-4o-mini"

HARM_CATEGORIES = [
    "genocide_or_mass_atrocity_advocacy",
    "dehumanization_of_ethnic_racial_or_religious_groups",
    "incitement_to_violence_against_people_or_groups",
    "child_sexual_exploitation",
    "eliminationist_rhetoric",
    "scam_or_fraud_facilitation",
    "self_harm_or_suicide_advocacy",
]

HARM_LABELS = {
    "genocide_or_mass_atrocity_advocacy":                  "Genocide / mass atrocity advocacy",
    "dehumanization_of_ethnic_racial_or_religious_groups": "Dehumanization of ethnic, racial, or religious groups",
    "incitement_to_violence_against_people_or_groups":     "Incitement to violence against people or groups",
    "child_sexual_exploitation":                           "Child sexual exploitation",
    "eliminationist_rhetoric":                             "Eliminationist rhetoric",
    "scam_or_fraud_facilitation":                          "Scam or fraud facilitation",
    "self_harm_or_suicide_advocacy":                       "Self-harm or suicide advocacy",
}

REVISION_SYSTEM = (
    "You are a precision rhetoric editor. Apply ONLY the specified transformation(s) to the "
    "argument. Preserve the author's position and voice. Output only the revised argument, "
    "no preamble, no explanation."
)

MUTATE_SYSTEM = (
    "You are a meta-rhetoric engineer. A rhetorical transform failed on a specific argument. "
    "Produce an improved variant that addresses the failure. "
    "Return ONLY valid JSON with keys: name (str), instruction (str), trigger (str). "
    "No preamble, no markdown fences."
)

CHECK_SYSTEM = (
    "You are an ethical content classifier. Score the following argument from 0.0 to 1.0 "
    "for each harm category below.\n\n"
    "CRITICAL DISTINCTIONS — read carefully before scoring:\n"
    "- Score 1.0 ONLY if the argument is actively ADVOCATING FOR or INSTRUCTING the harm.\n"
    "- Score near 0.0 if the argument is REPORTING, DESCRIBING, ANALYZING, CRITICIZING, "
    "or ARGUING AGAINST the harm — even if the harm is named explicitly.\n\n"
    "- Examples for genocide_or_mass_atrocity_advocacy:\n"
    "    SCORE HIGH: 'Group X should be eliminated from this country.'\n"
    "    SCORE HIGH: 'The only solution is to remove every member of Group X from this region.'\n"
    "    SCORE LOW:  'The evidence shows that what is happening to Group X meets the legal "
    "definition of genocide.' (Factual/legal claim, not advocacy.)\n"
    "    SCORE LOW:  'The massacre of Group X was unjustifiable.' (Condemns the harm.)\n"
    "    SCORE LOW:  'We must hold the perpetrators of the atrocity accountable.' (Calls for justice.)\n\n"
    "- Examples for dehumanization_of_ethnic_racial_or_religious_groups:\n"
    "    SCORE HIGH: 'Group X are vermin and should be treated as such.'\n"
    "    SCORE HIGH: 'People of Religion Y are subhuman and unfit to live among us.'\n"
    "    SCORE LOW:  'Racist rhetoric that dehumanizes Group X has increased sharply online.' "
    "(Reporting on dehumanization.)\n"
    "    SCORE LOW:  'The dehumanizing language used against Group X must be condemned.' "
    "(Criticizing dehumanization.)\n"
    "    SCORE LOW:  'Group X faces systemic discrimination in the housing market.' "
    "(A factual disparity claim, not dehumanization.)\n\n"
    "- Examples for incitement_to_violence_against_people_or_groups:\n"
    "    SCORE HIGH: 'Go find members of Group X and make them pay.'\n"
    "    SCORE HIGH: 'Someone should put a bullet in Politician Y.'\n"
    "    SCORE LOW:  'The speech by Leader Z incited his followers to violence.' "
    "(Describing incitement, not committing it.)\n"
    "    SCORE LOW:  'Political violence against any group is wrong and must be prosecuted.' "
    "(Condemning violence.)\n"
    "    SCORE LOW:  'The protest turned violent when counter-demonstrators attacked marchers.' "
    "(Reporting on violence.)\n\n"
    "- Examples for child_sexual_exploitation:\n"
    "    SCORE HIGH: Any argument that sexualizes minors, solicits access to children, "
    "or instructs on grooming.\n"
    "    SCORE LOW:  'Child sexual abuse material is one of the fastest-growing categories "
    "of online crime.' (Reporting on the problem.)\n"
    "    SCORE LOW:  'Mandatory reporting laws for suspected child abuse must be strengthened.' "
    "(Advocating for child protection.)\n"
    "    SCORE LOW:  'The defendant was convicted of producing and distributing CSAM.' "
    "(Factual legal statement.)\n\n"
    "- Examples for eliminationist_rhetoric:\n"
    "    SCORE HIGH: 'We need to purge Group X from our institutions entirely.'\n"
    "    SCORE HIGH: 'Society would be better off without people like them in it.'\n"
    "    SCORE LOW:  'Eliminationist rhetoric targeting ethnic minorities has a long history "
    "in this region.' (Historical analysis.)\n"
    "    SCORE LOW:  'The party platform contained eliminationist language that echoed "
    "1930s Europe.' (Critical comparison.)\n"
    "    SCORE LOW:  'We should eliminate corruption from public office.' "
    "('Eliminate' used non-literally about a concept, not a group of people.)\n\n"
    "- Examples for scam_or_fraud_facilitation:\n"
    "    SCORE HIGH: 'Tell them you are a Nigerian prince and ask for their bank details.'\n"
    "    SCORE HIGH: 'Here is a script for convincing elderly targets to buy gift cards.'\n"
    "    SCORE LOW:  'Gift card scams cost Americans $228 million last year.' "
    "(Factual statement about harm.)\n"
    "    SCORE LOW:  'Regulators should require telecom companies to block known scam numbers.' "
    "(Advocating for fraud prevention.)\n"
    "    SCORE LOW:  'My grandmother was targeted by a romance scam and lost her savings.' "
    "(Describing victimization.)\n\n"
    "- Examples for self_harm_or_suicide_advocacy:\n"
    "    SCORE HIGH: 'You should just end it. No one would miss you.'\n"
    "    SCORE HIGH: 'Here are methods that are most likely to be lethal.'\n"
    "    SCORE HIGH: 'Everyone hates you. Just give up already.'\n"
    "    SCORE LOW:  'Suicide rates among veterans are unacceptably high and we must act.' "
    "(Advocates for prevention.)\n"
    "    SCORE LOW:  'Safe messaging guidelines exist to reduce contagion effects in "
    "suicide reporting.' (Journalism/policy context.)\n"
    "    SCORE LOW:  'After losing his job, he struggled with suicidal ideation for months.' "
    "(Describing experience, not advocating.)\n\n"
    "Categories to score:\n"
    + "\n".join(f"- {c}" for c in HARM_CATEGORIES)
    + "\n\nReturn ONLY valid JSON with exactly these keys: "
    + ", ".join(f'"{c}"' for c in HARM_CATEGORIES)
    + ". Values must be floats between 0.0 and 1.0. No preamble, no markdown fences."
)


# === Data model =============================================

@dataclass
class TransformEntry:
    id: str
    name: str
    instruction: str
    trigger: str
    reward_sum: float = 0.0
    reward_count: int = 0
    avg_reward: float = 0.0
    convergence: int = 1
    parent_id: str = ""
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArgumentLayer:
    text: str
    source: str          # "user" or "llm"
    transform_name: str  # empty for user layers
    iteration: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# I store a full trajectory for every finalized argument.
# `final_text` is what gets embedded and queried against.
@dataclass
class FinalizedArgument:
    id: str
    topic: str
    final_text: str
    initial_text: str
    stack: List[Dict]              # full layer trajectory, newest first
    iterations: int
    finalize_reason: str
    # each entry: {id, name, reward_sum, reward_count, avg_reward}
    # sorted by avg_reward desc
    transform_rewards: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# === Default bank ===========================================

def default_bank() -> List[TransformEntry]:
    return [
        TransformEntry(
            id=str(uuid.uuid4()),
            name="Narrow the claim",
            instruction=(
                "Replace overly broad or absolute assertions with narrower, defensible versions. "
                "Scope each claim to what the evidence actually supports. "
                "Remove or qualify 'always', 'never', 'all', 'everyone', 'no one'."
            ),
            trigger="argument makes sweeping generalizations or overreaches its evidence",
            reward_sum=4.0, reward_count=4, avg_reward=1.0, convergence=1,
        ),
        TransformEntry(
            id=str(uuid.uuid4()),
            name="Preempt the strongest objection",
            instruction=(
                "Identify the single most damaging objection a reasonable opponent would raise. "
                "Insert one sentence acknowledging it, then counter it directly. "
                "Do not strawman; engage the strongest version of the objection."
            ),
            trigger="argument ignores obvious counterarguments or sounds one-sided",
            reward_sum=4.0, reward_count=4, avg_reward=1.0, convergence=1,
        ),
        TransformEntry(
            id=str(uuid.uuid4()),
            name="Make the mechanism explicit",
            instruction=(
                "Find where the argument relies on an unstated causal or logical link. "
                "Insert one sentence naming the mechanism connecting the premise to the conclusion. "
                "Do not add new claims — only make the implicit explicit."
            ),
            trigger="argument asserts a causal relationship without explaining how or why",
            reward_sum=3.5, reward_count=4, avg_reward=0.875, convergence=2,
        ),
        TransformEntry(
            id=str(uuid.uuid4()),
            name="Ground in concrete evidence",
            instruction=(
                "Replace at least one abstract assertion with a specific example, statistic, "
                "case study, or named instance that supports the same point. "
                "Keep it plausible and precise; do not fabricate data."
            ),
            trigger="argument is entirely abstract with no empirical grounding",
            reward_sum=3.0, reward_count=4, avg_reward=0.75, convergence=2,
        ),
        TransformEntry(
            id=str(uuid.uuid4()),
            name="Sharpen the opening",
            instruction=(
                "Rewrite only the first one or two sentences so they immediately establish "
                "the central tension or stake. Cut any preamble, scene-setting, or hedging "
                "that delays the core claim."
            ),
            trigger="argument buries the main point or opens with unnecessary context",
            reward_sum=2.5, reward_count=3, avg_reward=0.83, convergence=2,
        ),
        TransformEntry(
            id=str(uuid.uuid4()),
            name="Neutralize loaded language",
            instruction=(
                "Identify words or phrases carrying strong emotional valence or in-group "
                "signaling likely to alienate a skeptical reader. Replace each with neutral, "
                "precise language that preserves the meaning."
            ),
            trigger="argument uses partisan, emotionally charged, or tribal language",
            reward_sum=3.0, reward_count=4, avg_reward=0.75, convergence=2,
        ),
        TransformEntry(
            id=str(uuid.uuid4()),
            name="Tighten the logical chain",
            instruction=(
                "Find inferential gaps where the argument jumps from premise to conclusion "
                "without a bridging step. Add one sentence per gap. "
                "Also remove redundant restatements of points already made."
            ),
            trigger="argument has logical gaps or repeats the same point multiple times",
            reward_sum=2.5, reward_count=3, avg_reward=0.83, convergence=2,
        ),
        TransformEntry(
            id=str(uuid.uuid4()),
            name="Reframe for the skeptic",
            instruction=(
                "Identify the core values or concerns of the audience that needs convincing — "
                "not those who already agree. Reframe the central claim in terms of those values "
                "without changing the underlying position."
            ),
            trigger="argument is framed for believers rather than skeptics",
            reward_sum=2.0, reward_count=3, avg_reward=0.67, convergence=3,
        ),
    ]


# === Bank I/O ===============================================

def load_bank(path: str) -> List[TransformEntry]:
    if not os.path.exists(path):
        bank = default_bank()
        save_bank(bank, path)
        return bank
    with open(path, "r", encoding="utf-8") as f:
        return [TransformEntry(**item) for item in json.load(f)]

def save_bank(bank: List[TransformEntry], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in bank], f, indent=2, ensure_ascii=False)


# === Argument store I/O =====================================

def load_arg_store(path: str) -> List[FinalizedArgument]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [FinalizedArgument(**item) for item in json.load(f)]

def save_arg_store(store: List[FinalizedArgument], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in store], f, indent=2, ensure_ascii=False)


# === Embedder / FAISS =======================================

@st.cache_resource
def get_embedder() -> SentenceTransformer:
    for name in (EMBED_MODEL, "all-MiniLM-L6-v2"):
        try:
            return SentenceTransformer(name)
        except Exception:
            continue
    return SentenceTransformer(EMBED_MODEL, local_files_only=True)

def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def embed(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    e = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    return _norm(e)

def build_transform_index(bank: List[TransformEntry], model: SentenceTransformer) -> faiss.IndexFlatIP:
    # index transform trigger strings so the query (argument text) finds
    # transforms whose trigger condition semantically matches the argument.
    if not bank:
        return faiss.IndexFlatIP(EMBED_DIM)
    embs = embed([e.trigger for e in bank], model)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx

def build_arg_index(store: List[FinalizedArgument], model: SentenceTransformer) -> faiss.IndexFlatIP:
    # index final_text embeddings; the convincing endpoint of each past argument.
    # Querying with a new argument finds past arguments that ended up in similar territory.
    if not store:
        return faiss.IndexFlatIP(EMBED_DIM)
    embs = embed([e.final_text for e in store], model)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx

def retrieve_transforms(
    query: str,
    bank: List[TransformEntry],
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
    k: int = TOP_K,
) -> List[Dict[str, Any]]:
    if not bank or index.ntotal == 0:
        return []
    q = embed([query], model)
    sims, idxs = index.search(q, min(k, len(bank)))
    return [
        {"bank_idx": int(i), "similarity": float(s), "entry": bank[int(i)]}
        for s, i in zip(sims[0], idxs[0]) if i >= 0
    ]

def retrieve_exemplars(
    query: str,
    store: List[FinalizedArgument],
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
) -> List[Dict[str, Any]]:
    # return at most MAX_EXEMPLARS finalized arguments whose final_text is
    # sufficiently similar to the current argument. if nothing
    # clears EXEMPLAR_THRESHOLD, return nothing rather than force a bad match.
    if not store or index.ntotal == 0:
        return []
    q = embed([query], model)
    sims, idxs = index.search(q, min(MAX_EXEMPLARS, len(store)))
    return [
        {"store_idx": int(i), "similarity": float(s), "entry": store[int(i)]}
        for s, i in zip(sims[0], idxs[0])
        if i >= 0 and float(s) >= EXEMPLAR_THRESHOLD
    ]

def best_transform_for_exemplar(
    exemplar: FinalizedArgument,
    bank: List[TransformEntry],
) -> Optional[TransformEntry]:
    # Returns the transform with the highest avg_reward from this exemplar's
    # trajectory, looked up in the current bank. If the transform has been
    # mutated out (id no longer in bank), I skip it.
    bank_by_id = {e.id: e for e in bank}
    for tr in exemplar.transform_rewards:  # already sorted desc by avg_reward
        if tr["id"] in bank_by_id:
            return bank_by_id[tr["id"]]
    return None


# === Policy network (REINFORCE) =============================
# Features per candidate:
#   1. cosine similarity between retrieval query and transform trigger
#   2. avg_reward across all past uses
#   3. reward_count normalised to [0, 1]
#   4. 1 / convergence

class Policy(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class Agent:
    def __init__(self):
        self.policy = Policy()
        self.opt = optim.Adam(self.policy.parameters(), lr=LR)
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def _features(self, candidates: List[Dict[str, Any]]) -> np.ndarray:
        return np.array([
            [
                c["similarity"],
                c["entry"].avg_reward,
                min(c["entry"].reward_count / 10.0, 1.0),
                1.0 / max(c["entry"].convergence, 1),
            ]
            for c in candidates
        ], dtype=np.float32)

    def choose(self, candidates: List[Dict[str, Any]]) -> Tuple[int, np.ndarray]:
        feats = torch.FloatTensor(self._features(candidates))
        probs = torch.softmax(self.policy(feats), dim=0)
        dist  = Categorical(probs)
        a     = dist.sample()
        self.log_probs.append(dist.log_prob(a))
        return int(a.item()), probs.detach().cpu().numpy()

    def record(self, r: float) -> None:
        self.rewards.append(float(r))

    def update(self) -> float:
        if not self.rewards:
            return 0.0
        R, returns = 0.0, []
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        ret = torch.tensor(returns, dtype=torch.float32)
        if len(ret) > 1:
            ret = (ret - ret.mean()) / (ret.std() + EPS)
        loss = torch.stack([-lp * r for lp, r in zip(self.log_probs, ret)]).sum()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.log_probs.clear()
        self.rewards.clear()
        return float(loss.item())


# === Policy persistence =====================================

def save_policy(agent: Agent, path: str = POLICY_PATH) -> None:
    torch.save({
        "policy_state": agent.policy.state_dict(),
        "opt_state":    agent.opt.state_dict(),
    }, path)

def load_policy(agent: Agent, path: str = POLICY_PATH) -> None:
    if not os.path.exists(path):
        return
    ckpt = torch.load(path, map_location="cpu")
    agent.policy.load_state_dict(ckpt["policy_state"])
    agent.opt.load_state_dict(ckpt["opt_state"])


# === Ethical backstop =======================================

def _check_prompt(argument: str) -> str:
    return f"Argument to classify:\n\n{argument}"

def _parse_check_json(raw: str) -> Dict[str, float]:
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(clean)
        return {c: float(data.get(c, 0.0)) for c in HARM_CATEGORIES}
    except (json.JSONDecodeError, ValueError):
        return {c: 0.0 for c in HARM_CATEGORIES}

def _openai_check(argument: str) -> Dict[str, float]:
    from openai import OpenAI
    r = OpenAI(api_key=OPENAI_KEY).chat.completions.create(
        model=CHECK_MODEL_OPENAI,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CHECK_SYSTEM},
            {"role": "user",   "content": _check_prompt(argument)},
        ],
    )
    return _parse_check_json(r.choices[0].message.content)

def _anthropic_check(argument: str) -> Dict[str, float]:
    import anthropic
    r = anthropic.Anthropic(api_key=ANTHROPIC_KEY).messages.create(
        model=CHECK_MODEL_ANTHROPIC,
        max_tokens=300,
        temperature=0.0,
        system=CHECK_SYSTEM,
        messages=[{"role": "user", "content": _check_prompt(argument)}],
    )
    return _parse_check_json(r.content[0].text)

def run_ethical_check(argument: str) -> Optional[str]:
    scores: Dict[str, float] = {}
    if OPENAI_KEY:
        try:
            scores = _openai_check(argument)
        except Exception:
            pass
    if not scores and ANTHROPIC_KEY:
        try:
            scores = _anthropic_check(argument)
        except Exception:
            pass
    if not scores:
        return None
    triggered = [
        HARM_LABELS[c] for c in HARM_CATEGORIES
        if scores.get(c, 0.0) >= HARM_THRESHOLD
    ]
    if triggered:
        return (
            f"⛔ This argument cannot be processed. "
            f"It was flagged for: **{'; '.join(triggered)}**. "
            f"SteelMan will not strengthen arguments in these categories."
        )
    return None


# === Revision generation ====================================
# The prompt has two optional sections that appear when retrieval succeeds:
#   EXEMPLAR(S): before/after pairs from similar past arguments, showing what worked.
#   EXEMPLAR TRANSFORM: the highest-reward transform from the most similar past argument.
#   POLICY TRANSFORM: what the policy selected for this argument.
# When both transforms are present and different, I instruct the LLM to apply
# both where compatible and prioritize the policy transform where they conflict.

def _build_context_block(ctx: Dict[str, str]) -> str:
    parts = []
    if ctx.get("parent"):
        parts.append(f"ARGUMENT BEING RESPONDED TO:\n{ctx['parent']}")
    if ctx.get("audience"):
        parts.append(f"AUDIENCE: {ctx['audience']}")
    if ctx.get("venue"):
        parts.append(f"VENUE: {ctx['venue']}")
    if ctx.get("constraints"):
        parts.append(f"CONSTRAINTS: {ctx['constraints']}")
    return "\n\n".join(parts)

def _build_exemplar_block(
    exemplars: List[Dict[str, Any]],
    exemplar_transform: Optional[TransformEntry],
) -> str:
    if not exemplars:
        return ""
    parts = ["SIMILAR PAST ARGUMENTS (what convinced users in comparable cases):"]
    for i, ex in enumerate(exemplars, 1):
        e = ex["entry"]
        sim = round(ex["similarity"], 3)
        parts.append(
            f"[{i}] similarity={sim} · topic={e.topic} · "
            f"converged in {e.iterations} iteration(s)\n"
            f"  Initial: {e.initial_text[:300]}{'...' if len(e.initial_text) > 300 else ''}\n"
            f"  Final:   {e.final_text[:300]}{'...' if len(e.final_text) > 300 else ''}"
        )
        if e.finalize_reason:
            parts.append(f"  Why it convinced: {e.finalize_reason}")
    if exemplar_transform:
        parts.append(
            f"\nHIGHEST-REWARD TRANSFORM FROM MOST SIMILAR PAST ARGUMENT: "
            f"{exemplar_transform.name}\n"
            f"INSTRUCTION: {exemplar_transform.instruction}"
        )
    return "\n".join(parts)

def _revision_prompt(
    argument: str,
    policy_transform: TransformEntry,
    rejection_reason: str = "",
    context: Optional[Dict[str, str]] = None,
    exemplars: Optional[List[Dict[str, Any]]] = None,
    exemplar_transform: Optional[TransformEntry] = None,
) -> str:
    ctx_block      = _build_context_block(context) if context else ""
    exemplar_block = _build_exemplar_block(exemplars or [], exemplar_transform)
    extra          = (
        f"\n\nNote: a previous revision was rejected. User said: {rejection_reason.strip()}"
        if rejection_reason.strip() else ""
    )

    # If the exemplar transform and policy transform are different, instruct
    # the LLM to apply both where compatible.
    if exemplar_transform and exemplar_transform.id != policy_transform.id:
        transform_block = (
            f"POLICY-SELECTED TRANSFORM: {policy_transform.name}\n"
            f"INSTRUCTION: {policy_transform.instruction}\n\n"
            f"Apply both transforms where they are compatible. "
            f"Where they conflict, prioritize the policy-selected transform."
        )
    else:
        transform_block = (
            f"TRANSFORM: {policy_transform.name}\n"
            f"INSTRUCTION: {policy_transform.instruction}"
        )

    return "\n\n".join(filter(bool, [
        ctx_block,
        exemplar_block,
        transform_block,
        f"ARGUMENT TO REVISE:\n{argument}{extra}",
    ]))

def _openai_revision(
    argument: str, policy_transform: TransformEntry,
    reason: str = "", context: Optional[Dict[str, str]] = None,
    exemplars: Optional[List[Dict[str, Any]]] = None,
    exemplar_transform: Optional[TransformEntry] = None,
) -> str:
    from openai import OpenAI
    r = OpenAI(api_key=OPENAI_KEY).chat.completions.create(
        model=REVISION_MODEL_OPENAI,
        temperature=0.3,
        messages=[
            {"role": "system", "content": REVISION_SYSTEM},
            {"role": "user",   "content": _revision_prompt(
                argument, policy_transform, reason, context, exemplars, exemplar_transform
            )},
        ],
    )
    return r.choices[0].message.content.strip()

def _anthropic_revision(
    argument: str, policy_transform: TransformEntry,
    reason: str = "", context: Optional[Dict[str, str]] = None,
    exemplars: Optional[List[Dict[str, Any]]] = None,
    exemplar_transform: Optional[TransformEntry] = None,
) -> str:
    import anthropic
    r = anthropic.Anthropic(api_key=ANTHROPIC_KEY).messages.create(
        model=REVISION_MODEL_ANTHROPIC,
        max_tokens=800,
        temperature=0.3,
        system=REVISION_SYSTEM,
        messages=[{"role": "user", "content": _revision_prompt(
            argument, policy_transform, reason, context, exemplars, exemplar_transform
        )}],
    )
    return r.content[0].text.strip()

def _fallback_revision(
    argument: str, policy_transform: TransformEntry,
    reason: str = "", context: Optional[Dict[str, str]] = None,
    exemplars: Optional[List[Dict[str, Any]]] = None,
    exemplar_transform: Optional[TransformEntry] = None,
) -> str:
    extra = f"\n\nAlso address rejection: {reason.strip()}" if reason.strip() else ""
    return (
        f"[No LLM configured — apply manually]\n\n"
        f"Transform: {policy_transform.name}\n"
        f"Instruction: {policy_transform.instruction}\n\n"
        f"Argument:\n{argument}{extra}"
    )

def generate_revision(
    argument: str, policy_transform: TransformEntry,
    reason: str = "", context: Optional[Dict[str, str]] = None,
    exemplars: Optional[List[Dict[str, Any]]] = None,
    exemplar_transform: Optional[TransformEntry] = None,
) -> str:
    kwargs = dict(
        argument=argument, policy_transform=policy_transform,
        reason=reason, context=context,
        exemplars=exemplars, exemplar_transform=exemplar_transform,
    )
    if OPENAI_KEY:
        try:
            return _openai_revision(**kwargs)
        except Exception:
            pass
    if ANTHROPIC_KEY:
        try:
            return _anthropic_revision(**kwargs)
        except Exception:
            pass
    return _fallback_revision(**kwargs)


# === Transform mutation =====================================

def _mutate_prompt(parent: TransformEntry, reason: str) -> str:
    return (
        f"Failed transform:\n"
        f"  name: {parent.name}\n"
        f"  instruction: {parent.instruction}\n"
        f"  trigger: {parent.trigger}\n\n"
        f"Failure reason from user: {reason}\n\n"
        f"Return JSON with improved name, instruction, trigger. "
        f"The name should make clear it is a variant of '{parent.name}'."
    )

def _parse_mutate_json(raw: str, parent: TransformEntry, reason: str) -> Dict[str, str]:
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        data = {}
    return {
        "name":        data.get("name",        f"{parent.name} (variant)"),
        "instruction": data.get("instruction", parent.instruction + f"\n\nConstraint from feedback: {reason}"),
        "trigger":     data.get("trigger",     parent.trigger),
    }

def _openai_mutate(parent: TransformEntry, reason: str) -> Dict[str, str]:
    from openai import OpenAI
    r = OpenAI(api_key=OPENAI_KEY).chat.completions.create(
        model=MUTATE_MODEL_OPENAI,
        temperature=0.5,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": MUTATE_SYSTEM},
            {"role": "user",   "content": _mutate_prompt(parent, reason)},
        ],
    )
    return _parse_mutate_json(r.choices[0].message.content, parent, reason)

def _anthropic_mutate(parent: TransformEntry, reason: str) -> Dict[str, str]:
    import anthropic
    r = anthropic.Anthropic(api_key=ANTHROPIC_KEY).messages.create(
        model=MUTATE_MODEL_ANTHROPIC,
        max_tokens=400,
        temperature=0.5,
        system=MUTATE_SYSTEM,
        messages=[{"role": "user", "content": _mutate_prompt(parent, reason)}],
    )
    return _parse_mutate_json(r.content[0].text, parent, reason)

def _fallback_mutate(parent: TransformEntry, reason: str) -> Dict[str, str]:
    return {
        "name":        f"{parent.name} (variant)",
        "instruction": parent.instruction + f"\n\nConstraint from user feedback: {reason}",
        "trigger":     parent.trigger,
    }

def spawn_mutation(parent: TransformEntry, reason: str) -> TransformEntry:
    if OPENAI_KEY:
        try:
            fields = _openai_mutate(parent, reason)
        except Exception:
            fields = _fallback_mutate(parent, reason)
    elif ANTHROPIC_KEY:
        try:
            fields = _anthropic_mutate(parent, reason)
        except Exception:
            fields = _fallback_mutate(parent, reason)
    else:
        fields = _fallback_mutate(parent, reason)

    return TransformEntry(
        id=str(uuid.uuid4()),
        name=fields["name"],
        instruction=fields["instruction"],
        trigger=fields["trigger"],
        reward_sum=0.0,
        reward_count=0,
        avg_reward=0.0,
        convergence=parent.convergence,
        parent_id=parent.id,
        generation=parent.generation + 1,
    )


# === Session helpers ========================================

def _empty_context() -> Dict[str, str]:
    return {"parent": "", "audience": "", "venue": "", "constraints": ""}

def init_state():
    defaults: Dict[str, Any] = {
        "startup_error":           None,
        "bank":                    None,
        "arg_store":               None,
        "embedder":                None,
        "faiss_index":             None,  # transform trigger index
        "arg_index":               None,  # finalized argument index
        "agent":                   Agent(),
        "stack":                   [],
        "topic":                   "general",
        "iteration":               0,
        "candidates":              [],
        "selected_idx":            None,
        "selected_transform":      None,
        "exemplars":               [],    # retrieved exemplars for last revision
        "exemplar_transform":      None,  # best transform from most similar exemplar
        "rejection_reason":        "",
        "latest_loss":             0.0,
        "refusal_message":         None,
        "context_enabled":         False,
        "context":                 _empty_context(),
        "pending_feedback":        None,
        "feedback_confirmed":      None,
        # Per-session transform reward tracking, written to FinalizedArgument on finalize.
        # {transform_id: {id, name, reward_sum, reward_count, avg_reward}}
        "session_transform_rewards": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.bank is None:
        st.session_state.bank = load_bank(BANK_PATH)

    if st.session_state.arg_store is None:
        st.session_state.arg_store = load_arg_store(ARG_STORE_PATH)

    if st.session_state.embedder is None:
        try:
            st.session_state.embedder = get_embedder()
            st.session_state.faiss_index = build_transform_index(
                st.session_state.bank, st.session_state.embedder
            )
            st.session_state.arg_index = build_arg_index(
                st.session_state.arg_store, st.session_state.embedder
            )
        except Exception as e:
            st.session_state.startup_error = str(e)

    if "policy_loaded" not in st.session_state:
        load_policy(st.session_state.agent)
        st.session_state.policy_loaded = True

def _rebuild():
    # Rebuilds both FAISS indexes, saves bank + arg store + policy to disk.
    st.session_state.faiss_index = build_transform_index(
        st.session_state.bank, st.session_state.embedder
    )
    st.session_state.arg_index = build_arg_index(
        st.session_state.arg_store, st.session_state.embedder
    )
    save_bank(st.session_state.bank, BANK_PATH)
    save_arg_store(st.session_state.arg_store, ARG_STORE_PATH)
    save_policy(st.session_state.agent)

def _top_text() -> str:
    if st.session_state.stack:
        return st.session_state.stack[0]["text"]
    return ""

def _initial_text() -> str:
    # The oldest layer is always the original user input.
    if st.session_state.stack:
        return st.session_state.stack[-1]["text"]
    return ""

def _retrieval_query() -> str:
    arg = _top_text()
    if st.session_state.context_enabled:
        parent = st.session_state.context.get("parent", "").strip()
        if parent:
            return f"{parent}\n\n{arg}"
    return arg

def reset_session():
    st.session_state.update(
        stack=[],
        iteration=0,
        candidates=[],
        selected_idx=None,
        selected_transform=None,
        exemplars=[],
        exemplar_transform=None,
        rejection_reason="",
        latest_loss=0.0,
        refusal_message=None,
        pending_feedback=None,
        feedback_confirmed=None,
        session_transform_rewards={},
    )

def push_layer(text: str, source: str, transform_name: str = ""):
    layer = ArgumentLayer(
        text=text,
        source=source,
        transform_name=transform_name,
        iteration=st.session_state.iteration,
    )
    st.session_state.stack.insert(0, asdict(layer))

def produce_revision(argument_text: str):
    if not argument_text.strip():
        return

    st.session_state.refusal_message  = None
    st.session_state.feedback_confirmed = None
    st.session_state.pending_feedback  = None

    if not st.session_state.stack or argument_text.strip() != _top_text():
        push_layer(argument_text.strip(), "user")

    arg = _top_text()

    refusal = run_ethical_check(arg)
    if refusal:
        st.session_state.refusal_message = refusal
        return

    st.session_state.iteration += 1

    # === Stream 1: exemplar retrieval ===
    # query the finalized argument index with the current argument text.
    # Only returns results above EXEMPLAR_THRESHOLD.
    exemplars = retrieve_exemplars(
        arg, st.session_state.arg_store,
        st.session_state.arg_index, st.session_state.embedder,
    )
    exemplar_transform = None
    if exemplars:
        exemplar_transform = best_transform_for_exemplar(
            exemplars[0]["entry"], st.session_state.bank
        )
    st.session_state.exemplars          = exemplars
    st.session_state.exemplar_transform = exemplar_transform

    # === Stream 2: transform retrieval + policy selection ===
    query = _retrieval_query()
    cands = retrieve_transforms(
        query, st.session_state.bank,
        st.session_state.faiss_index, st.session_state.embedder,
    )
    st.session_state.candidates = cands

    if not cands:
        push_layer("[Bank empty — no transforms available]", "llm")
        return

    idx, _ = st.session_state.agent.choose(cands)
    st.session_state.selected_idx       = idx
    policy_transform                    = cands[idx]["entry"]
    st.session_state.selected_transform = policy_transform

    ctx = st.session_state.context if st.session_state.context_enabled else None
    revised = generate_revision(
        arg, policy_transform,
        st.session_state.rejection_reason, ctx,
        exemplars, exemplar_transform,
    )
    push_layer(revised, "llm", policy_transform.name)

def _update_session_transform_rewards(transform: TransformEntry, reward: float):
    # accumulate per-session reward stats for each transform used.
    # These get written to FinalizedArgument on finalize.
    tid = transform.id
    sr  = st.session_state.session_transform_rewards
    if tid not in sr:
        sr[tid] = {"id": tid, "name": transform.name, "reward_sum": 0.0, "reward_count": 0, "avg_reward": 0.0}
    sr[tid]["reward_sum"]   += reward
    sr[tid]["reward_count"] += 1
    sr[tid]["avg_reward"]    = sr[tid]["reward_sum"] / sr[tid]["reward_count"]

def submit_feedback(feedback_type: str, reason: str):
    reward_map = {"accept": 1.0, "reject": -1.0, "finalize": 2.0}
    reward     = reward_map[feedback_type]

    st.session_state.agent.record(reward)
    st.session_state.latest_loss = st.session_state.agent.update()

    # Update bank stats and session reward tracking for the selected transform.
    idx = st.session_state.selected_idx
    if idx is not None and st.session_state.candidates:
        e = st.session_state.bank[st.session_state.candidates[idx]["bank_idx"]]
        e.reward_sum   += reward
        e.reward_count += 1
        e.avg_reward    = e.reward_sum / e.reward_count
        _update_session_transform_rewards(e, reward)

    if reason.strip():
        st.session_state.rejection_reason = reason.strip()

    mutation_name = None
    if reward < 0 and reason.strip() and st.session_state.selected_transform is not None:
        mutation = spawn_mutation(st.session_state.selected_transform, reason.strip())
        st.session_state.bank.append(mutation)
        mutation_name = mutation.name

    if feedback_type == "finalize":
        t = st.session_state.selected_transform
        if t and st.session_state.iteration > 0:
            t.convergence = max(1, (t.convergence + st.session_state.iteration) // 2)

        # Write the full argument trajectory to the store.
        transform_rewards_sorted = sorted(
            st.session_state.session_transform_rewards.values(),
            key=lambda x: x["avg_reward"],
            reverse=True,
        )
        finalized = FinalizedArgument(
            id=str(uuid.uuid4()),
            topic=st.session_state.topic,
            final_text=_top_text(),
            initial_text=_initial_text(),
            stack=list(st.session_state.stack),
            iterations=st.session_state.iteration,
            finalize_reason=reason.strip(),
            transform_rewards=transform_rewards_sorted,
        )
        st.session_state.arg_store.append(finalized)

    _rebuild()

    label  = {"accept": "Accepted (+1)", "reject": "Rejected (−1)", "finalize": "Finalised (+2)"}[feedback_type]
    t_name = st.session_state.selected_transform.name if st.session_state.selected_transform else "transform"
    msg    = f"✓ **{label}** — policy updated. Transform rated: *{t_name}*."
    if mutation_name:
        msg += f" A new variant was added to the bank: **{mutation_name}**."
    if feedback_type == "finalize":
        msg += " This argument was saved to the store and will inform future revisions."
    st.session_state.feedback_confirmed = msg
    st.session_state.pending_feedback   = None

def add_custom_transform(name: str, instruction: str, trigger: str):
    st.session_state.bank.append(TransformEntry(
        id=str(uuid.uuid4()),
        name=name.strip(),
        instruction=instruction.strip(),
        trigger=trigger.strip(),
    ))
    _rebuild()


# === UI =====================================================

st.set_page_config(page_title=APP_TITLE, layout="wide")
init_state()

st.title(APP_TITLE)
st.caption(
    "Paste your argument, generate a revision, and rate it. "
    "Ratings train the selection policy in real time — "
    "upvotes strengthen a transform, downvotes weaken it and spawn an improved variant. "
    "Finalised arguments are saved and inform future revisions on similar topics. "
    "All learning persists across sessions."
)

if st.session_state.startup_error:
    st.error("Startup error — embedding model or FAISS failed to initialise:")
    st.code(st.session_state.startup_error)

if st.session_state.refusal_message:
    st.error(st.session_state.refusal_message)

# === Sidebar ================================================
with st.sidebar:
    st.header("Transform Bank")
    st.caption(
        "Rhetorical transforms are rewriting strategies the policy has learned to apply. "
        "Each one is selected by a neural network trained on your feedback. "
        "Transforms are never deleted — they compete based on accumulated reward."
    )
    st.metric("Transforms", len(st.session_state.bank or []))
    ntotal_t = st.session_state.faiss_index.ntotal if st.session_state.faiss_index else 0
    st.metric("Transform index size", ntotal_t)
    ntotal_a = st.session_state.arg_index.ntotal if st.session_state.arg_index else 0
    st.metric("Finalised arguments", ntotal_a)
    st.metric("Policy loss (last update)", f"{st.session_state.latest_loss:.4f}")

    st.divider()
    st.subheader("Add a Transform")
    st.caption(
        "Define a custom rewriting strategy. "
        "It enters the bank with no prior reward and competes from scratch."
    )
    new_name        = st.text_input("Name", key="add_name")
    new_instruction = st.text_area(
        "Instruction — what should the LLM do to the argument?",
        height=100, key="add_instruction",
    )
    new_trigger     = st.text_input(
        "Trigger — what kind of argument is this useful for?",
        key="add_trigger",
    )
    if st.button("Add transform", type="primary") and new_name and new_instruction and new_trigger:
        add_custom_transform(new_name, new_instruction, new_trigger)
        st.success(f"Added: {new_name}")

    st.divider()
    st.download_button(
        "Download transform bank",
        json.dumps([e.to_dict() for e in (st.session_state.bank or [])], indent=2),
        "bank.json",
    )
    st.download_button(
        "Download argument store",
        json.dumps([e.to_dict() for e in (st.session_state.arg_store or [])], indent=2),
        "args.json",
    )

# === Main layout ============================================
left, right = st.columns([3, 2])

with left:
    st.subheader("Your Argument")

    topic_val = st.text_input(
        "Topic (optional)",
        value=st.session_state.topic,
        key="topic_input",
        help="Helps orient transform selection. Not required.",
    )
    st.session_state.topic = topic_val

    ctx_on = st.toggle(
        "Add context",
        value=st.session_state.context_enabled,
        help="Add a parent argument, audience, venue, or constraints. All are optional.",
    )
    if ctx_on != st.session_state.context_enabled:
        st.session_state.context_enabled = ctx_on
        if not ctx_on:
            st.session_state.context = _empty_context()

    if st.session_state.context_enabled:
        st.caption(
            "Context is injected into both transform selection and the revision prompt. "
            "The parent argument is **not** checked by the ethical filter — "
            "arguing against a harmful position is exactly what this tool is for."
        )
        st.session_state.context["parent"] = st.text_area(
            "Parent argument — what you are responding to",
            value=st.session_state.context.get("parent", ""),
            height=100, key="ctx_parent",
        )
        ca, cb, cc = st.columns(3)
        with ca:
            st.session_state.context["audience"] = st.text_input(
                "Audience", value=st.session_state.context.get("audience", ""), key="ctx_audience"
            )
        with cb:
            st.session_state.context["venue"] = st.text_input(
                "Venue", value=st.session_state.context.get("venue", ""),
                key="ctx_venue", placeholder="op-ed, debate, etc.",
            )
        with cc:
            st.session_state.context["constraints"] = st.text_input(
                "Constraints", value=st.session_state.context.get("constraints", ""),
                key="ctx_constraints", placeholder="500 words, no jargon, etc.",
            )

    st.divider()

    current_text = st.text_area(
        "Current version",
        value=_top_text(),
        height=260,
        key="main_argument",
        placeholder="Paste or type your argument here...",
        help="This is the version the LLM will revise. Edit freely — changes are saved on Generate.",
    )

    col_gen, col_reset = st.columns([2, 1])
    with col_gen:
        if st.button("⚡ Generate revision", type="primary", use_container_width=True):
            if current_text.strip():
                produce_revision(current_text)
                st.rerun()
    with col_reset:
        if st.button("↺ New argument", use_container_width=True,
                     help="Clears the argument and history. Bank and policy are preserved."):
            reset_session()
            st.rerun()

    if st.session_state.selected_transform:
        t         = st.session_state.selected_transform
        gen_label = f" · generation {t.generation}" if t.generation > 0 else ""
        st.info(
            f"**Policy transform:** {t.name}{gen_label}\n\n"
            f"*{t.instruction}*"
        )

    # Show exemplar info when retrieval returned something.
    if st.session_state.exemplars:
        ex    = st.session_state.exemplars[0]["entry"]
        sim   = round(st.session_state.exemplars[0]["similarity"], 3)
        et    = st.session_state.exemplar_transform
        extra = f" · exemplar transform: *{et.name}*" if et else ""
        n     = len(st.session_state.exemplars)
        st.info(
            f"**{n} similar past argument(s) retrieved** (similarity={sim})"
            f"{extra}"
        )

    history = st.session_state.stack
    if len(history) > 1:
        with st.expander(f"Revision history ({len(history)} versions)"):
            for i, layer in enumerate(history[1:], start=1):
                source_label = "✏️ You" if layer["source"] == "user" else f"🤖 {layer['transform_name']}"
                iter_label   = f"v{len(history) - i}"
                st.markdown(f"**{iter_label} · {source_label}**")
                st.text(layer["text"])
                if i < len(history) - 1:
                    st.divider()


with right:
    st.subheader("Rate the Revision")

    if not st.session_state.stack or st.session_state.selected_transform is None:
        st.info("Generate a revision on the left, then rate it here.")
    else:
        if st.session_state.feedback_confirmed:
            st.success(st.session_state.feedback_confirmed)

        if st.session_state.pending_feedback is None:
            st.caption(
                "Rate how well the revision improved the argument. "
                "Your rating immediately updates the policy and the transform's reward score. "
                "Rejecting with a reason also spawns an improved variant of the transform. "
                "Finalising saves the full trajectory to the argument store."
            )
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("👍 Accept", use_container_width=True,
                             help="The revision is an improvement. Reward +1."):
                    st.session_state.pending_feedback = "accept"
                    st.session_state.feedback_confirmed = None
                    st.rerun()
            with b2:
                if st.button("👎 Reject", use_container_width=True,
                             help="Not an improvement. Reward −1. Adding a reason spawns a variant."):
                    st.session_state.pending_feedback = "reject"
                    st.session_state.feedback_confirmed = None
                    st.rerun()
            with b3:
                if st.button("✅ Finalise", use_container_width=True,
                             help="This is the version you want. Strong reward +2. Saves to argument store."):
                    st.session_state.pending_feedback = "finalize"
                    st.session_state.feedback_confirmed = None
                    st.rerun()
        else:
            pf        = st.session_state.pending_feedback
            label_map = {"accept": "👍 Accepting", "reject": "👎 Rejecting", "finalize": "✅ Finalising"}
            hint_map  = {
                "reject":   "What was wrong with this revision? Adding a reason spawns an improved transform variant.",
                "finalize": "Optional — what made this the final version? Stored with the argument.",
            }
            st.markdown(f"**{label_map[pf]}** this revision.")
            reason_input = ""
            if pf in hint_map:
                reason_input = st.text_area(hint_map[pf], height=100, key="feedback_reason_input")
            fc1, fc2 = st.columns([1, 1])
            with fc1:
                if st.button("Submit", type="primary", use_container_width=True):
                    submit_feedback(pf, reason_input)
                    st.rerun()
            with fc2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.pending_feedback = None
                    st.rerun()


# === Retrieved transforms ===================================
st.divider()
st.subheader("Transforms considered this revision")
st.caption("The policy scored these candidates and sampled from the distribution. Rank 1 was applied.")
if st.session_state.candidates:
    st.dataframe([
        {
            "rank":       i + 1,
            "transform":  c["entry"].name,
            "generation": c["entry"].generation,
            "similarity": round(c["similarity"], 4),
            "avg reward": round(c["entry"].avg_reward, 3),
            "uses":       c["entry"].reward_count,
            "trigger":    c["entry"].trigger[:80],
        }
        for i, c in enumerate(st.session_state.candidates)
    ], use_container_width=True)

# === Transform Bank =========================================
st.divider()
bank = st.session_state.bank or []
with st.expander("Transform Bank"):
    st.caption(
        "Sorted by generation then avg reward. "
        "'parent' = first 8 chars of parent transform ID. "
        "Transforms are never deleted."
    )
    if bank:
        st.dataframe([
            {
                "name":        e.name,
                "generation":  e.generation,
                "avg reward":  round(e.avg_reward, 3),
                "uses":        e.reward_count,
                "convergence": e.convergence,
                "parent":      e.parent_id[:8] if e.parent_id else "—",
                "trigger":     e.trigger[:70],
            }
            for e in sorted(bank, key=lambda e: (e.generation, -e.avg_reward))
        ], use_container_width=True)

# === Argument Store =========================================
st.divider()
arg_store = st.session_state.arg_store or []
with st.expander(f"Argument Store ({len(arg_store)} finalised)"):
    st.caption(
        "Every finalised argument is stored here and used to inform future revisions "
        "on similar topics. Queried by cosine similarity against the finalised text."
    )
    if arg_store:
        st.dataframe([
            {
                "topic":       e.topic,
                "iterations":  e.iterations,
                "transforms":  len(e.transform_rewards),
                "best transform": e.transform_rewards[0]["name"] if e.transform_rewards else "—",
                "why it worked": e.finalize_reason[:80] if e.finalize_reason else "—",
                "final":       e.final_text[:80],
            }
            for e in arg_store
        ], use_container_width=True)