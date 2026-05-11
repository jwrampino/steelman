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
# I keep all tuneable knobs here so nothing is buried in logic.

load_dotenv()

APP_TITLE = "SteelMan"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BANK_PATH   = "steelman_bank.json"   # where I persist the transform library
POLICY_PATH = "steelman_policy.pt"  # where I persist the learned network weights

EMBED_DIM  = 384   # output dim of all-MiniLM-L6-v2
TOP_K      = 8     # candidates I retrieve per query
HIDDEN_DIM = 64    # policy network hidden layer width
LR         = 1e-3  # Adam learning rate
GAMMA      = 1.0   # discount factor; 1.0 = no discounting (single-step episodes)
EPS        = 1e-8  # numerical stability floor for return normalisation

# I check both providers and use whichever key is present; OpenAI takes priority.
OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Swap these for whatever models you're targeting.
REVISION_MODEL_ANTHROPIC = "claude-haiku-4-5-20251001"
MUTATE_MODEL_ANTHROPIC   = "claude-haiku-4-5-20251001"
REVISION_MODEL_OPENAI    = "gpt-4o-mini"
MUTATE_MODEL_OPENAI      = "gpt-4o-mini"

# I use a tight system prompt for revisions: apply the transform, nothing else.
REVISION_SYSTEM = (
    "You are a precision rhetoric editor. Apply ONLY the specified transformation to the "
    "argument. Preserve the author's position and voice. Output only the revised argument, "
    "no preamble, no explanation."
)

# For mutations I want structured output, so I ask explicitly for JSON.
MUTATE_SYSTEM = (
    "You are a meta-rhetoric engineer. A rhetorical transform failed on a specific argument. "
    "Produce an improved variant that addresses the failure. "
    "Return ONLY valid JSON with keys: name (str), instruction (str), trigger (str). "
    "No preamble, no markdown fences."
)


# === Data model =============================================
# I represent each rhetorical transform as a single entry in the bank.
# `instruction` is what I send to the LLM at revision time.
# `trigger` is what I embed for semantic retrieval — it describes when this
# transform is useful, so the cosine search can match it against the argument.
# `parent_id` and `generation` track mutation lineage; originals have both empty/0.
# I never delete entries — only append and accumulate reward signal.

@dataclass
class TransformEntry:
    id: str
    name: str
    instruction: str
    trigger: str
    reward_sum: float = 0.0
    reward_count: int = 0
    avg_reward: float = 0.0
    convergence: int = 1   # rolling avg iterations it took the user to accept this transform
    parent_id: str = ""    # id of the transform I mutated from; empty for originals
    generation: int = 0    # how many mutations deep I am from the original

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# === Default bank ===========================================
# I seed the bank with eight archetypal transforms covering the most common
# failure modes in persuasive arguments. Each gets a small positive prior
# so the policy has something to work with before real user feedback arrives.

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
# I write the full bank on every feedback event, so the file is always
# current and a restart loses nothing.

def load_bank(path: str) -> List[TransformEntry]:
    if not os.path.exists(path):
        # first run — seed with defaults and persist immediately
        bank = default_bank()
        save_bank(bank, path)
        return bank
    with open(path, "r", encoding="utf-8") as f:
        return [TransformEntry(**item) for item in json.load(f)]

def save_bank(bank: List[TransformEntry], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in bank], f, indent=2, ensure_ascii=False)


# === Embedder / FAISS =======================================
# I embed each transform's `trigger` string and index them with inner-product
# search (cosine similarity on unit vectors). At query time I embed the
# argument text and find the transforms whose trigger conditions are most
# semantically similar to what the argument is doing.

@st.cache_resource
def get_embedder() -> SentenceTransformer:
    # I try the full HF repo name first, then the short alias, then local cache.
    for name in (EMBED_MODEL, "all-MiniLM-L6-v2"):
        try:
            return SentenceTransformer(name)
        except Exception:
            continue
    return SentenceTransformer(EMBED_MODEL, local_files_only=True)

def _norm(x: np.ndarray) -> np.ndarray:
    # L2-normalise rows so inner product == cosine similarity
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def embed(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    e = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    return _norm(e)

def build_index(bank: List[TransformEntry], model: SentenceTransformer) -> faiss.IndexFlatIP:
    # I rebuild the index from scratch on every bank mutation — cheap enough
    # at this scale, and avoids stale vectors after mutations or additions.
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
    # I return up to k candidates with their bank position, similarity score,
    # and the full entry so the policy network can read all four features.
    if not bank or index.ntotal == 0:
        return []
    q = embed([query], model)
    sims, idxs = index.search(q, min(k, len(bank)))
    return [
        {"bank_idx": int(i), "similarity": float(s), "entry": bank[int(i)]}
        for s, i in zip(sims[0], idxs[0]) if i >= 0
    ]


# === Policy network (REINFORCE) =============================
# I score each candidate with a small MLP, then sample from the softmax
# distribution. This lets me explore while still biasing toward historically
# good transforms. The four input features are:
#   1. cosine similarity between argument and transform trigger
#   2. avg_reward of the transform across all past uses
#   3. reward_count normalised to [0, 1] (proxy for confidence)
#   4. 1 / convergence (faster-converging transforms score higher)

class Policy(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),  # scalar score per candidate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class Agent:
    def __init__(self):
        self.policy = Policy()
        self.opt = optim.Adam(self.policy.parameters(), lr=LR)
        # I accumulate log-probs and rewards within an episode (one argument
        # session), then flush them on update().
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def _features(self, candidates: List[Dict[str, Any]]) -> np.ndarray:
        return np.array([
            [
                c["similarity"],
                c["entry"].avg_reward,
                min(c["entry"].reward_count / 10.0, 1.0),  # cap at 1
                1.0 / max(c["entry"].convergence, 1),
            ]
            for c in candidates
        ], dtype=np.float32)

    def choose(self, candidates: List[Dict[str, Any]]) -> Tuple[int, np.ndarray]:
        # I score all candidates, convert to a probability distribution, and
        # sample — stochastic selection lets low-scoring transforms occasionally
        # get picked and earn their way up (or confirm they should stay low).
        feats = torch.FloatTensor(self._features(candidates))
        probs = torch.softmax(self.policy(feats), dim=0)
        dist  = Categorical(probs)
        a     = dist.sample()
        self.log_probs.append(dist.log_prob(a))
        return int(a.item()), probs.detach().cpu().numpy()

    def record(self, r: float) -> None:
        self.rewards.append(float(r))

    def update(self) -> float:
        # Standard REINFORCE: compute discounted returns, normalise them,
        # then backprop -log_prob * return for each step in the episode.
        # I flush both buffers afterward so the next episode starts clean.
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
# I save both the network weights and the optimiser state so momentum
# and adaptive learning rates carry over across restarts.

def save_policy(agent: Agent, path: str = POLICY_PATH) -> None:
    torch.save({
        "policy_state": agent.policy.state_dict(),
        "opt_state":    agent.opt.state_dict(),
    }, path)

def load_policy(agent: Agent, path: str = POLICY_PATH) -> None:
    if not os.path.exists(path):
        return  # first run — nothing to load
    ckpt = torch.load(path, map_location="cpu")
    agent.policy.load_state_dict(ckpt["policy_state"])
    agent.opt.load_state_dict(ckpt["opt_state"])


# === Revision generation ====================================
# I build a tightly scoped prompt: name the transform, give the instruction,
# hand over the argument. The system prompt forbids the model from doing
# anything other than applying the transform, which keeps revisions surgical.

def _revision_prompt(argument: str, transform: TransformEntry, rejection_reason: str = "") -> str:
    extra = (
        f"\n\nNote: a previous revision was rejected. User said: {rejection_reason.strip()}"
        if rejection_reason.strip() else ""
    )
    return (
        f"TRANSFORMATION: {transform.name}\n"
        f"INSTRUCTION: {transform.instruction}\n\n"
        f"ARGUMENT TO REVISE:\n{argument}{extra}"
    )

def _openai_revision(argument: str, transform: TransformEntry, reason: str = "") -> str:
    from openai import OpenAI
    r = OpenAI(api_key=OPENAI_KEY).chat.completions.create(
        model=REVISION_MODEL_OPENAI,
        temperature=0.3,  # low temp — I want consistent application, not creativity
        messages=[
            {"role": "system", "content": REVISION_SYSTEM},
            {"role": "user",   "content": _revision_prompt(argument, transform, reason)},
        ],
    )
    return r.choices[0].message.content.strip()

def _anthropic_revision(argument: str, transform: TransformEntry, reason: str = "") -> str:
    import anthropic
    r = anthropic.Anthropic(api_key=ANTHROPIC_KEY).messages.create(
        model=REVISION_MODEL_ANTHROPIC,
        max_tokens=800,
        temperature=0.3,
        system=REVISION_SYSTEM,
        messages=[{"role": "user", "content": _revision_prompt(argument, transform, reason)}],
    )
    return r.content[0].text.strip()

def _fallback_revision(argument: str, transform: TransformEntry, reason: str = "") -> str:
    # No LLM available — I surface the transform instructions so the user can
    # apply them manually. Honest about what's missing.
    extra = f"\n\nAlso address rejection: {reason.strip()}" if reason.strip() else ""
    return (
        f"[No LLM configured — apply manually]\n\n"
        f"Transform: {transform.name}\n"
        f"Instruction: {transform.instruction}\n\n"
        f"Original:\n{argument}{extra}"
    )

def generate_revision(argument: str, transform: TransformEntry, reason: str = "") -> str:
    # I try OpenAI first, then Anthropic, then fall back to the manual template.
    if OPENAI_KEY:
        try:
            return _openai_revision(argument, transform, reason)
        except Exception:
            pass
    if ANTHROPIC_KEY:
        try:
            return _anthropic_revision(argument, transform, reason)
        except Exception:
            pass
    return _fallback_revision(argument, transform, reason)


# === Transform mutation =====================================
# When the user rejects a revision with an explanation, I spawn a mutant
# variant of the transform that failed. The mutation is itself an LLM call:
# I pass the parent transform plus the failure reason and ask for an improved
# name, instruction, and trigger. The mutant enters the bank at generation+1
# with zeroed reward stats — it competes against its parent from scratch.
# I never delete the parent; both survive and the policy decides which wins.

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
    # I strip markdown fences defensively — some models wrap JSON anyway.
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        data = {}
    # Fall back field-by-field so a partial parse still produces a valid entry.
    return {
        "name":        data.get("name",        f"{parent.name} (variant)"),
        "instruction": data.get("instruction", parent.instruction + f"\n\nConstraint from feedback: {reason}"),
        "trigger":     data.get("trigger",     parent.trigger),
    }

def _openai_mutate(parent: TransformEntry, reason: str) -> Dict[str, str]:
    from openai import OpenAI
    r = OpenAI(api_key=OPENAI_KEY).chat.completions.create(
        model=MUTATE_MODEL_OPENAI,
        temperature=0.5,  # slightly higher than revision — I want some variation in mutations
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
    # Without an LLM I can still produce a valid mutation by appending the
    # user's feedback as an explicit constraint on the instruction.
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
        convergence=parent.convergence,  # inherit parent's convergence estimate as prior
        parent_id=parent.id,
        generation=parent.generation + 1,
    )


# === Session helpers ========================================
# Streamlit reruns the whole script on every interaction, so I keep all
# mutable state in st.session_state. init_state() is idempotent — it only
# sets keys that don't already exist, so reruns are safe.

def init_state():
    defaults: Dict[str, Any] = {
        "startup_error":      None,
        "bank":               None,   # loaded below
        "embedder":           None,   # loaded below
        "faiss_index":        None,   # built below
        "agent":              Agent(),
        "argument":           "",
        "topic":              "general",
        "iteration":          0,      # how many revisions this session
        "candidates":         [],     # last retrieved candidate list
        "selected_idx":       None,   # index into candidates that was chosen
        "selected_transform": None,   # the TransformEntry that was applied
        "revision":           "",     # latest LLM output
        "rejection_reason":   "",     # last rejection text; injected into next prompt
        "latest_loss":        0.0,    # most recent REINFORCE loss (for display)
        "last_mutation":      None,   # name of the last spawned mutation (for toast)
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

    # I load saved policy weights exactly once per browser session,
    # guarded by a flag so reruns don't clobber in-progress learning.
    if "policy_loaded" not in st.session_state:
        load_policy(st.session_state.agent)
        st.session_state.policy_loaded = True

def _rebuild():
    # I call this after every bank mutation. It re-embeds all trigger strings,
    # rebuilds the FAISS index, and flushes both the bank JSON and the policy
    # checkpoint to disk so nothing is lost on restart.
    st.session_state.faiss_index = build_index(
        st.session_state.bank, st.session_state.embedder
    )
    save_bank(st.session_state.bank, BANK_PATH)
    save_policy(st.session_state.agent)

def reset_session(argument: str, topic: str):
    # I clear per-argument state but leave the bank, embedder, and agent intact.
    st.session_state.update(
        argument=argument.strip(),
        topic=topic.strip() or "general",
        iteration=0,
        candidates=[],
        selected_idx=None,
        selected_transform=None,
        revision="",
        rejection_reason="",
        latest_loss=0.0,
        last_mutation=None,
    )

def produce_revision():
    arg = st.session_state.argument
    if not arg:
        return
    st.session_state.iteration += 1
    st.session_state.last_mutation = None
    # I retrieve the top-k transforms whose triggers are most similar to the
    # argument, then let the policy choose among them stochastically.
    cands = retrieve(
        arg, st.session_state.bank,
        st.session_state.faiss_index, st.session_state.embedder,
    )
    st.session_state.candidates = cands
    if not cands:
        st.session_state.revision = "[Bank empty]"
        return
    idx, _ = st.session_state.agent.choose(cands)
    st.session_state.selected_idx = idx
    transform = cands[idx]["entry"]
    st.session_state.selected_transform = transform
    st.session_state.revision = generate_revision(
        arg, transform, st.session_state.rejection_reason
    )

def apply_feedback(reward: float, reason: str = ""):
    # I do three things here:
    #   1. Run a REINFORCE gradient step on the policy network.
    #   2. Update avg_reward on the selected transform in the bank.
    #   3. If the feedback is negative and the user gave a reason,
    #      spawn a mutant variant and add it to the bank.
    st.session_state.agent.record(reward)
    st.session_state.latest_loss = st.session_state.agent.update()

    idx = st.session_state.selected_idx
    if idx is not None and st.session_state.candidates:
        e = st.session_state.bank[st.session_state.candidates[idx]["bank_idx"]]
        e.reward_sum  += reward
        e.reward_count += 1
        e.avg_reward   = e.reward_sum / e.reward_count

    if reason.strip():
        st.session_state.rejection_reason = reason.strip()

    if reward < 0 and reason.strip() and st.session_state.selected_transform is not None:
        mutation = spawn_mutation(st.session_state.selected_transform, reason.strip())
        st.session_state.bank.append(mutation)
        st.session_state.last_mutation = mutation.name

    _rebuild()

def finalize(reason: str):
    # Finalize gives a +2 reward (stronger signal than a plain accept),
    # then updates the transform's convergence estimate so it knows roughly
    # how many iterations it typically takes to get accepted.
    apply_feedback(2.0, reason)
    t = st.session_state.selected_transform
    if t and st.session_state.iteration > 0:
        t.convergence = max(1, (t.convergence + st.session_state.iteration) // 2)
    _rebuild()

def add_custom_transform(name: str, instruction: str, trigger: str):
    # Manual additions enter at generation 0 with no parent and zeroed stats —
    # same starting position as a mutant, same competition rules.
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
st.caption("Iterative argument rewriting · transform retrieval · REINFORCE · mutation on rejection")

if st.session_state.startup_error:
    st.error("Startup error:")
    st.code(st.session_state.startup_error)

if st.session_state.last_mutation:
    st.info(f"⚗️ Mutation spawned and added to bank: **{st.session_state.last_mutation}**")

# === Sidebar ================================================
with st.sidebar:
    st.header("Bank")
    st.metric("Transforms", len(st.session_state.bank or []))
    ntotal = st.session_state.faiss_index.ntotal if st.session_state.faiss_index else 0
    st.metric("FAISS vectors", ntotal)
    st.metric("Last policy loss", f"{st.session_state.latest_loss:.4f}")

    st.divider()

    with st.expander("Add transform"):
        new_name        = st.text_input("Name",                     key="add_name")
        new_instruction = st.text_area("Instruction", height=100,   key="add_instruction")
        new_trigger     = st.text_input("Trigger (when to use this)", key="add_trigger")
        if st.button("Add") and new_name and new_instruction and new_trigger:
            add_custom_transform(new_name, new_instruction, new_trigger)
            st.success(f"Added: {new_name}")

    st.divider()

    if st.button("Reset bank to defaults"):
        # I wipe both the bank and the policy checkpoint so everything restarts clean.
        st.session_state.bank = default_bank()
        st.session_state.agent = Agent()
        st.session_state.policy_loaded = True  # prevent immediate reload of old checkpoint
        if os.path.exists(POLICY_PATH):
            os.remove(POLICY_PATH)
        _rebuild()
    st.download_button(
        "Download bank JSON",
        json.dumps([e.to_dict() for e in (st.session_state.bank or [])], indent=2),
        "bank.json",
    )

# === Main columns ===========================================
left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    topic    = st.text_input("Topic",    value=st.session_state.topic)
    argument = st.text_area("Argument", value=st.session_state.argument, height=280)
    c1, c2   = st.columns(2)
    with c1:
        if st.button("Reset session"):
            reset_session(argument, topic)
    with c2:
        if st.button("Generate revision"):
            if argument.strip():
                # If the text changed since last session, start fresh.
                if argument.strip() != st.session_state.argument:
                    reset_session(argument, topic)
                produce_revision()
    st.caption(f"Iteration: {st.session_state.iteration}")
    if st.session_state.selected_transform:
        t = st.session_state.selected_transform
        gen_label = f" · g{t.generation}" if t.generation > 0 else ""
        st.info(f"**{t.name}**{gen_label}\n\n{t.instruction}")

with right:
    st.subheader("Revision")
    if st.session_state.revision:
        st.text_area("Output", value=st.session_state.revision, height=280)
        reason = st.text_input(
            "Reason (required to trigger mutation on reject)",
            value=st.session_state.rejection_reason,
        )
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("👍 Accept"):
                apply_feedback(1.0)
        with b2:
            if st.button("👎 Reject"):
                # A rejection without a reason still penalises the transform
                # but does not spawn a mutation — I need the reason to know
                # what to improve.
                apply_feedback(-1.0, reason)
        with b3:
            if st.button("✅ Finalize"):
                finalize(reason or "Accepted.")
    else:
        st.info("No revision yet.")

# === Retrieved candidates ===================================
st.divider()
st.subheader("Retrieved transforms (this iteration)")
if st.session_state.candidates:
    st.dataframe([
        {
            "rank":       i + 1,
            "transform":  c["entry"].name,
            "gen":        c["entry"].generation,
            "similarity": round(c["similarity"], 4),
            "avg_reward": round(c["entry"].avg_reward, 3),
            "n":          c["entry"].reward_count,
            "trigger":    c["entry"].trigger[:80],
        }
        for i, c in enumerate(st.session_state.candidates)
    ])

# === Full bank ==============================================
st.divider()
bank = st.session_state.bank or []
with st.expander(f"Full transform bank ({len(bank)} entries)"):
    if bank:
        # I sort by generation first so lineages are readable, then by avg_reward
        # descending so the best transforms within each generation appear first.
        st.dataframe([
            {
                "name":        e.name,
                "gen":         e.generation,
                "avg_reward":  round(e.avg_reward, 3),
                "n":           e.reward_count,
                "convergence": e.convergence,
                "parent":      e.parent_id[:8] if e.parent_id else "—",
                "trigger":     e.trigger[:70],
            }
            for e in sorted(bank, key=lambda e: (e.generation, -e.avg_reward))
        ])
