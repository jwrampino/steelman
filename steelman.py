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
POLICY_PATH = "steelman_policy.pt"

EMBED_DIM  = 384
TOP_K      = 8
HIDDEN_DIM = 64
LR         = 1e-3
GAMMA      = 1.0
EPS        = 1e-8

# A single category scoring at or above this threshold is a hard block.
# Set high (0.8) to avoid false positives on academic discussion, journalism,
# or arguments that describe/refute a harmful position without advocating it.
HARM_THRESHOLD = 0.8

OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

REVISION_MODEL_ANTHROPIC = "claude-haiku-4-5-20251001"
MUTATE_MODEL_ANTHROPIC   = "claude-haiku-4-5-20251001"
CHECK_MODEL_ANTHROPIC    = "claude-haiku-4-5-20251001"
REVISION_MODEL_OPENAI    = "gpt-4o-mini"
MUTATE_MODEL_OPENAI      = "gpt-4o-mini"
CHECK_MODEL_OPENAI       = "gpt-4o-mini"

# Seven hard categories. Evaluated against the user's argument text only —
# never the parent, since arguing *against* these positions is a legitimate
# and often important use of this tool.
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
    "You are a precision rhetoric editor. Apply ONLY the specified transformation to the "
    "argument. Preserve the author's position and voice. Output only the revised argument, "
    "no preamble, no explanation."
)

MUTATE_SYSTEM = (
    "You are a meta-rhetoric engineer. A rhetorical transform failed on a specific argument. "
    "Produce an improved variant that addresses the failure. "
    "Return ONLY valid JSON with keys: name (str), instruction (str), trigger (str). "
    "No preamble, no markdown fences."
)

# I am very explicit here about the distinction between advocacy and
# description/analysis/refutation, because vague prompts produce false positives
# on arguments that merely discuss or argue against a harmful position.
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
    instruction: str   # directive sent to the LLM at revision time
    trigger: str       # embedded for FAISS retrieval; describes when this transform is useful
    reward_sum: float = 0.0
    reward_count: int = 0
    avg_reward: float = 0.0
    convergence: int = 1   # rolling avg iterations before user acceptance
    parent_id: str = ""    # empty for originals
    generation: int = 0    # 0 for originals; +1 per mutation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# A single version of the argument.
# source = "user" means typed/pasted by the user.
# source = "llm"  means generated by a transform.
@dataclass
class ArgumentLayer:
    text: str
    source: str
    transform_name: str   # empty for user layers
    iteration: int

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

def build_index(bank: List[TransformEntry], model: SentenceTransformer) -> faiss.IndexFlatIP:
    if not bank:
        return faiss.IndexFlatIP(EMBED_DIM)
    embs = embed([e.trigger for e in bank], model)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx

def retrieve(
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


# === Policy network (REINFORCE) =============================
# Four input features per candidate:
#   1. cosine similarity between retrieval query and transform trigger
#   2. avg_reward across all past uses of this transform
#   3. reward_count normalised to [0, 1] (confidence proxy)
#   4. 1 / convergence (faster-converging transforms score higher)

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
# Runs on the user's argument text before every revision attempt.
# The parent argument is never checked — arguing against harmful positions
# is a legitimate and important use of this tool.
# Fails open if no LLM is available (returns None = pass).

def _check_prompt(argument: str) -> str:
    return f"Argument to classify:\n\n{argument}"

def _parse_check_json(raw: str) -> Dict[str, float]:
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(clean)
        return {c: float(data.get(c, 0.0)) for c in HARM_CATEGORIES}
    except (json.JSONDecodeError, ValueError):
        # Malformed response — fail open rather than block a legitimate argument.
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
    # Returns None if the argument passes.
    # Returns a refusal string naming the triggered category if it fails.
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
        cats = "; ".join(triggered)
        return (
            f"⛔ This argument cannot be processed. "
            f"It was flagged for: **{cats}**. "
            f"SteelMan will not strengthen arguments in these categories."
        )
    return None


# === Revision generation ====================================

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

def _revision_prompt(
    argument: str,
    transform: TransformEntry,
    rejection_reason: str = "",
    context: Optional[Dict[str, str]] = None,
) -> str:
    ctx_block = _build_context_block(context) if context else ""
    extra = (
        f"\n\nNote: a previous revision was rejected. User said: {rejection_reason.strip()}"
        if rejection_reason.strip() else ""
    )
    return "\n\n".join(filter(bool, [
        ctx_block,
        f"TRANSFORMATION: {transform.name}\nINSTRUCTION: {transform.instruction}",
        f"ARGUMENT TO REVISE:\n{argument}{extra}",
    ]))

def _openai_revision(
    argument: str, transform: TransformEntry,
    reason: str = "", context: Optional[Dict[str, str]] = None,
) -> str:
    from openai import OpenAI
    r = OpenAI(api_key=OPENAI_KEY).chat.completions.create(
        model=REVISION_MODEL_OPENAI,
        temperature=0.3,
        messages=[
            {"role": "system", "content": REVISION_SYSTEM},
            {"role": "user",   "content": _revision_prompt(argument, transform, reason, context)},
        ],
    )
    return r.choices[0].message.content.strip()

def _anthropic_revision(
    argument: str, transform: TransformEntry,
    reason: str = "", context: Optional[Dict[str, str]] = None,
) -> str:
    import anthropic
    r = anthropic.Anthropic(api_key=ANTHROPIC_KEY).messages.create(
        model=REVISION_MODEL_ANTHROPIC,
        max_tokens=800,
        temperature=0.3,
        system=REVISION_SYSTEM,
        messages=[{"role": "user", "content": _revision_prompt(argument, transform, reason, context)}],
    )
    return r.content[0].text.strip()

def _fallback_revision(
    argument: str, transform: TransformEntry,
    reason: str = "", context: Optional[Dict[str, str]] = None,
) -> str:
    extra = f"\n\nAlso address rejection: {reason.strip()}" if reason.strip() else ""
    return (
        f"[No LLM configured — apply manually]\n\n"
        f"Transform: {transform.name}\n"
        f"Instruction: {transform.instruction}\n\n"
        f"Argument:\n{argument}{extra}"
    )

def generate_revision(
    argument: str, transform: TransformEntry,
    reason: str = "", context: Optional[Dict[str, str]] = None,
) -> str:
    if OPENAI_KEY:
        try:
            return _openai_revision(argument, transform, reason, context)
        except Exception:
            pass
    if ANTHROPIC_KEY:
        try:
            return _anthropic_revision(argument, transform, reason, context)
        except Exception:
            pass
    return _fallback_revision(argument, transform, reason, context)


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
        "startup_error":      None,
        "bank":               None,
        "embedder":           None,
        "faiss_index":        None,
        "agent":              Agent(),
        # stack: list of ArgumentLayer dicts, newest at index 0.
        # The LLM always operates on stack[0].
        "stack":              [],
        "topic":              "general",
        "iteration":          0,
        "candidates":         [],
        "selected_idx":       None,
        "selected_transform": None,
        "rejection_reason":   "",
        "latest_loss":        0.0,
        "refusal_message":    None,
        "context_enabled":    False,
        "context":            _empty_context(),
        # Feedback flow state:
        # pending_feedback = None | "accept" | "reject" | "finalize"
        # When set, the UI shows a reason input + Submit button instead of the three rating buttons.
        "pending_feedback":   None,
        # confirmation message shown after feedback is submitted
        "feedback_confirmed": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.bank is None:
        st.session_state.bank = load_bank(BANK_PATH)

    if st.session_state.embedder is None:
        try:
            st.session_state.embedder = get_embedder()
            st.session_state.faiss_index = build_index(
                st.session_state.bank, st.session_state.embedder
            )
        except Exception as e:
            st.session_state.startup_error = str(e)

    if "policy_loaded" not in st.session_state:
        load_policy(st.session_state.agent)
        st.session_state.policy_loaded = True

def _rebuild():
    st.session_state.faiss_index = build_index(
        st.session_state.bank, st.session_state.embedder
    )
    save_bank(st.session_state.bank, BANK_PATH)
    save_policy(st.session_state.agent)

def _top_text() -> str:
    if st.session_state.stack:
        return st.session_state.stack[0]["text"]
    return ""

def _retrieval_query() -> str:
    # Prepend parent to argument when context is on, so transform selection
    # is aware of the dialectical situation.
    arg = _top_text()
    if st.session_state.context_enabled:
        parent = st.session_state.context.get("parent", "").strip()
        if parent:
            return f"{parent}\n\n{arg}"
    return arg

def reset_session():
    # Clears the current argument and revision history.
    # Bank, policy weights, and transform history are preserved.
    st.session_state.update(
        stack=[],
        iteration=0,
        candidates=[],
        selected_idx=None,
        selected_transform=None,
        rejection_reason="",
        latest_loss=0.0,
        refusal_message=None,
        pending_feedback=None,
        feedback_confirmed=None,
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
    # argument_text is whatever is currently in the main text area.
    # If it differs from the top of the stack, push it as a user layer first.
    if not argument_text.strip():
        return

    st.session_state.refusal_message = None
    st.session_state.feedback_confirmed = None
    st.session_state.pending_feedback = None

    # Sync the text area to the stack.
    if not st.session_state.stack or argument_text.strip() != _top_text():
        push_layer(argument_text.strip(), "user")

    arg = _top_text()

    # Ethical backstop — check user's argument only, never the parent.
    refusal = run_ethical_check(arg)
    if refusal:
        st.session_state.refusal_message = refusal
        return

    st.session_state.iteration += 1

    query = _retrieval_query()
    cands = retrieve(query, st.session_state.bank, st.session_state.faiss_index, st.session_state.embedder)
    st.session_state.candidates = cands

    if not cands:
        push_layer("[Bank empty — no transforms available]", "llm")
        return

    idx, _ = st.session_state.agent.choose(cands)
    st.session_state.selected_idx = idx
    transform = cands[idx]["entry"]
    st.session_state.selected_transform = transform

    ctx = st.session_state.context if st.session_state.context_enabled else None
    revised = generate_revision(arg, transform, st.session_state.rejection_reason, ctx)
    push_layer(revised, "llm", transform.name)

def submit_feedback(feedback_type: str, reason: str):
    # Called when the user clicks Submit in the two-step feedback form.
    reward_map = {"accept": 1.0, "reject": -1.0, "finalize": 2.0}
    reward = reward_map[feedback_type]

    st.session_state.agent.record(reward)
    st.session_state.latest_loss = st.session_state.agent.update()

    idx = st.session_state.selected_idx
    if idx is not None and st.session_state.candidates:
        e = st.session_state.bank[st.session_state.candidates[idx]["bank_idx"]]
        e.reward_sum   += reward
        e.reward_count += 1
        e.avg_reward    = e.reward_sum / e.reward_count

    if reason.strip():
        st.session_state.rejection_reason = reason.strip()

    mutation_name = None
    if reward < 0 and reason.strip() and st.session_state.selected_transform is not None:
        mutation = spawn_mutation(st.session_state.selected_transform, reason.strip())
        st.session_state.bank.append(mutation)
        mutation_name = mutation.name

    if feedback_type == "finalize" and st.session_state.selected_transform is not None:
        t = st.session_state.selected_transform
        if st.session_state.iteration > 0:
            t.convergence = max(1, (t.convergence + st.session_state.iteration) // 2)

    _rebuild()

    # Build the confirmation message shown to the user.
    label = {"accept": "Accepted (+1)", "reject": "Rejected (−1)", "finalize": "Finalised (+2)"}[feedback_type]
    t_name = st.session_state.selected_transform.name if st.session_state.selected_transform else "transform"
    msg = f"✓ **{label}** — policy updated. Transform rated: *{t_name}*."
    if mutation_name:
        msg += f" A new variant was added to the bank: **{mutation_name}**."
    st.session_state.feedback_confirmed = msg
    st.session_state.pending_feedback = None

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
    ntotal = st.session_state.faiss_index.ntotal if st.session_state.faiss_index else 0
    st.metric("FAISS index size", ntotal)
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
        "Download bank as JSON",
        json.dumps([e.to_dict() for e in (st.session_state.bank or [])], indent=2),
        "bank.json",
    )

# === Main layout ============================================
left, right = st.columns([3, 2])

with left:
    st.subheader("Your Argument")

    # Topic
    topic_val = st.text_input(
        "Topic (optional)",
        value=st.session_state.topic,
        key="topic_input",
        help="Helps orient transform selection. Not required.",
    )
    st.session_state.topic = topic_val

    # Context toggle
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

    # === Main argument text area ============================
    # Always shows the current top of the stack (or empty on first load).
    # The user can edit it freely — on clicking Generate, the current text
    # is synced to the stack as a user layer if it has changed.

    current_text = st.text_area(
        "Current version",
        value=_top_text(),
        height=260,
        key="main_argument",
        placeholder="Paste or type your argument here...",
        help=(
            "This is the version the LLM will revise. "
            "Edit it freely. When you click Generate, any changes are saved automatically."
        ),
    )

    col_gen, col_reset = st.columns([2, 1])
    with col_gen:
        if st.button("⚡ Generate revision", type="primary", use_container_width=True):
            if current_text.strip():
                produce_revision(current_text)
                st.rerun()
    with col_reset:
        if st.button("↺ New argument", use_container_width=True,
                     help="Clears the argument and revision history. Your transform bank and policy are preserved."):
            reset_session()
            st.rerun()

    # === Active transform label =============================
    if st.session_state.selected_transform:
        t = st.session_state.selected_transform
        gen_label = f" · generation {t.generation}" if t.generation > 0 else ""
        st.info(
            f"**Last transform applied:** {t.name}{gen_label}\n\n"
            f"*{t.instruction}*"
        )

    # === Revision history ===================================
    history = st.session_state.stack
    if len(history) > 1:
        with st.expander(f"Revision history ({len(history)} versions)"):
            # Show from index 1 onward (index 0 is displayed in the text area above).
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
        # === Feedback confirmed message =====================
        if st.session_state.feedback_confirmed:
            st.success(st.session_state.feedback_confirmed)

        # === Two-step feedback flow =========================
        # Step 1: three rating buttons.
        # Step 2: after clicking one, a reason input + Submit appears.
        # This makes it clear that (a) your click was registered,
        # (b) you can optionally explain why, and (c) the submission is explicit.

        if st.session_state.pending_feedback is None:
            st.caption(
                "Rate how well the revision improved the argument. "
                "Your rating immediately updates the policy and the transform's reward score. "
                "Rejecting with a reason also spawns an improved variant of the transform."
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
                             help="The revision is not an improvement. Reward −1. Adding a reason spawns a variant."):
                    st.session_state.pending_feedback = "reject"
                    st.session_state.feedback_confirmed = None
                    st.rerun()
            with b3:
                if st.button("✅ Finalise", use_container_width=True,
                             help="This is the version you want. Strong reward +2."):
                    st.session_state.pending_feedback = "finalize"
                    st.session_state.feedback_confirmed = None
                    st.rerun()

        else:
            # Step 2: reason input.
            pf = st.session_state.pending_feedback
            label_map = {
                "accept":   "👍 Accepting",
                "reject":   "👎 Rejecting",
                "finalize": "✅ Finalising",
            }
            hint_map = {
                "accept":   "Optional — what worked? (does not affect policy directly, but logged)",
                "reject":   "What was wrong with this revision? Adding a reason spawns an improved transform variant.",
                "finalize": "Optional — what made this the final version?",
            }
            st.markdown(f"**{label_map[pf]}** this revision.")
            reason_input = st.text_area(
                hint_map[pf],
                height=100,
                key="feedback_reason_input",
            )
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
st.caption(
    "The policy scored each of these candidates and sampled from the resulting distribution. "
    "Rank 1 is the transform that was applied."
)
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

# === Full bank ==============================================
st.divider()
bank = st.session_state.bank or []
with st.expander("Transform Bank"):
    st.caption(
        "All transforms sorted by generation then avg reward. "
        "'parent' shows the first 8 chars of the parent transform's ID. "
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