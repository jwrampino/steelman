import os
import json
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st

# faiss
try:
    import faiss
except ImportError:
    raise ImportError("Please install faiss-cpu or faiss-gpu.")

# embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install sentence-transformers.")

# RL
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
except ImportError:
    raise ImportError("Please install torch.")

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

APP_TITLE = "SteelMan Prototype"
# FIX: Use the full HF repository name to avoid OSError
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
BANK_PATH = "steelman_bank.json"
EMBED_DIM = 384  
TOP_K_RETRIEVAL = 8
POLICY_HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
GAMMA = 1.0  
EPS = 1e-8

# Optional API settings:
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ------------------------------------------------------------
# DATA MODEL
# ------------------------------------------------------------

@dataclass
class ArgumentEntry:
    id: str
    text: str
    topic: str
    reward_sum: float
    reward_count: int
    avg_reward: float
    iterations_to_convergence: int
    convinced_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------
# DEFAULT BANK
# ------------------------------------------------------------

def default_bank() -> List[ArgumentEntry]:
    seed = [
        ArgumentEntry(
            id=str(uuid.uuid4()),
            text=(
                "Cities should reduce car dependency rather than simply 'ban cars.' "
                "Transportation is a major source of urban emissions, and policies like "
                "congestion pricing, protected bike lanes, and frequent public transit have "
                "reduced pollution in comparable cities without eliminating mobility."
            ),
            topic="urban policy",
            reward_sum=4.0,
            reward_count=5,
            avg_reward=0.8,
            iterations_to_convergence=2,
            convinced_reason="Specific, evidence-based, and less absolutist."
        ),
        ArgumentEntry(
            id=str(uuid.uuid4()),
            text=(
                "A persuasive argument is usually stronger when it anticipates the strongest "
                "objection. Instead of asserting that opponents are ignorant, explain why a "
                "reasonable person might disagree and then respond directly to that concern."
            ),
            topic="rhetoric",
            reward_sum=5.0,
            reward_count=5,
            avg_reward=1.0,
            iterations_to_convergence=1,
            convinced_reason="Balanced tone and strong counterargument handling."
        ),
        ArgumentEntry(
            id=str(uuid.uuid4()),
            text=(
                "If the goal is persuasion rather than self-expression, claims should be narrowed "
                "to what can actually be defended. Replace sweeping statements with precise claims, "
                "concrete evidence, and one clearly stated mechanism."
            ),
            topic="argumentation",
            reward_sum=4.0,
            reward_count=4,
            avg_reward=1.0,
            iterations_to_convergence=1,
            convinced_reason="More defensible and easier to follow."
        ),
        ArgumentEntry(
            id=str(uuid.uuid4()),
            text=(
                "Arguments often fail because they overreach. A narrower claim that admits limits "
                "can be more persuasive than a maximal claim that invites easy rebuttal."
            ),
            topic="argumentation",
            reward_sum=3.0,
            reward_count=4,
            avg_reward=0.75,
            iterations_to_convergence=2,
            convinced_reason="Measured tone felt more credible."
        ),
    ]
    return seed


# ------------------------------------------------------------
# BANK / FAISS HELPERS
# ------------------------------------------------------------

def load_bank(path: str) -> List[ArgumentEntry]:
    if not os.path.exists(path):
        bank = default_bank()
        save_bank(bank, path)
        return bank

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [ArgumentEntry(**item) for item in raw]


def save_bank(bank: List[ArgumentEntry], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([entry.to_dict() for entry in bank], f, indent=2, ensure_ascii=False)


import os
import json
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st

# faiss
try:
    import faiss
except ImportError:
    raise ImportError("Please install faiss-cpu or faiss-gpu.")

# embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install sentence-transformers.")

# RL
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
except ImportError:
    raise ImportError("Please install torch.")

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

APP_TITLE = "SteelMan Prototype"
# FIX: Use the full HF repository name to avoid OSError
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
BANK_PATH = "steelman_bank.json"
EMBED_DIM = 384  
TOP_K_RETRIEVAL = 8
POLICY_HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
GAMMA = 1.0  
EPS = 1e-8

# Optional API settings:
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ------------------------------------------------------------
# DATA MODEL
# ------------------------------------------------------------

@dataclass
class ArgumentEntry:
    id: str
    text: str
    topic: str
    reward_sum: float
    reward_count: int
    avg_reward: float
    iterations_to_convergence: int
    convinced_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------
# DEFAULT BANK
# ------------------------------------------------------------

def default_bank() -> List[ArgumentEntry]:
    seed = [
        ArgumentEntry(
            id=str(uuid.uuid4()),
            text=(
                "Cities should reduce car dependency rather than simply 'ban cars.' "
                "Transportation is a major source of urban emissions, and policies like "
                "congestion pricing, protected bike lanes, and frequent public transit have "
                "reduced pollution in comparable cities without eliminating mobility."
            ),
            topic="urban policy",
            reward_sum=4.0,
            reward_count=5,
            avg_reward=0.8,
            iterations_to_convergence=2,
            convinced_reason="Specific, evidence-based, and less absolutist."
        ),
        ArgumentEntry(
            id=str(uuid.uuid4()),
            text=(
                "A persuasive argument is usually stronger when it anticipates the strongest "
                "objection. Instead of asserting that opponents are ignorant, explain why a "
                "reasonable person might disagree and then respond directly to that concern."
            ),
            topic="rhetoric",
            reward_sum=5.0,
            reward_count=5,
            avg_reward=1.0,
            iterations_to_convergence=1,
            convinced_reason="Balanced tone and strong counterargument handling."
        ),
        ArgumentEntry(
            id=str(uuid.uuid4()),
            text=(
                "If the goal is persuasion rather than self-expression, claims should be narrowed "
                "to what can actually be defended. Replace sweeping statements with precise claims, "
                "concrete evidence, and one clearly stated mechanism."
            ),
            topic="argumentation",
            reward_sum=4.0,
            reward_count=4,
            avg_reward=1.0,
            iterations_to_convergence=1,
            convinced_reason="More defensible and easier to follow."
        ),
        ArgumentEntry(
            id=str(uuid.uuid4()),
            text=(
                "Arguments often fail because they overreach. A narrower claim that admits limits "
                "can be more persuasive than a maximal claim that invites easy rebuttal."
            ),
            topic="argumentation",
            reward_sum=3.0,
            reward_count=4,
            avg_reward=0.75,
            iterations_to_convergence=2,
            convinced_reason="Measured tone felt more credible."
        ),
    ]
    return seed


# ------------------------------------------------------------
# BANK / FAISS HELPERS
# ------------------------------------------------------------

def load_bank(path: str) -> List[ArgumentEntry]:
    if not os.path.exists(path):
        bank = default_bank()
        save_bank(bank, path)
        return bank

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [ArgumentEntry(**item) for item in raw]


def save_bank(bank: List[ArgumentEntry], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([entry.to_dict() for entry in bank], f, indent=2, ensure_ascii=False)


@st.cache_resource
def get_embedder() -> SentenceTransformer:
    # Try multiple identifiers to bypass the "not a valid model identifier" error
    try:
        return SentenceTransformer(EMBED_MODEL_NAME)
    except Exception:
        try:
            # Fallback to the short name
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            # Final attempt: force local only if it's already in your cache
            return SentenceTransformer(EMBED_MODEL_NAME, local_files_only=True)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embs = embs.astype(np.float32)
    embs = normalize_rows(embs)
    return embs


def build_faiss_index(bank: List[ArgumentEntry], model: SentenceTransformer) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    texts = [entry.text for entry in bank]
    if len(texts) == 0:
        index = faiss.IndexFlatIP(EMBED_DIM)
        return index, np.zeros((0, EMBED_DIM), dtype=np.float32)

    embs = embed_texts(texts, model)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, embs


def retrieve_candidates(
    query_text: str,
    bank: List[ArgumentEntry],
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
    top_k: int = TOP_K_RETRIEVAL,
) -> List[Dict[str, Any]]:
    if len(bank) == 0 or index.ntotal == 0:
        return []

    q = embed_texts([query_text], model)
    sims, idxs = index.search(q, min(top_k, len(bank)))
    sims = sims[0]
    idxs = idxs[0]

    out = []
    for sim, idx in zip(sims, idxs):
        if idx < 0: continue
        entry = bank[int(idx)]
        out.append({
            "bank_idx": int(idx),
            "similarity": float(sim),
            "entry": entry,
        })
    return out


# ------------------------------------------------------------
# POLICY NETWORK (REINFORCE-STYLE)
# ------------------------------------------------------------

class RetrievalPolicy(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = POLICY_HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        scores = self.fc3(x).squeeze(-1)  
        return scores


class RetrievalAgent:
    def __init__(self, learning_rate: float = LEARNING_RATE, gamma: float = GAMMA):
        self.policy = RetrievalPolicy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def candidate_features(self, candidates: List[Dict[str, Any]]) -> np.ndarray:
        feats = []
        for c in candidates:
            entry = c["entry"]
            similarity = c["similarity"]
            avg_reward = entry.avg_reward
            reward_count_norm = min(entry.reward_count / 10.0, 1.0)
            inv_iterations_norm = 1.0 / max(entry.iterations_to_convergence, 1)
            feats.append([
                similarity,
                avg_reward,
                reward_count_norm,
                inv_iterations_norm,
            ])
        return np.array(feats, dtype=np.float32)

    def choose_candidate(self, candidates: List[Dict[str, Any]]) -> Tuple[int, np.ndarray]:
        feats = self.candidate_features(candidates)
        feats_tensor = torch.FloatTensor(feats)

        scores = self.policy(feats_tensor)
        probs = torch.softmax(scores, dim=0)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.log_probs.append(log_prob)

        return int(action.item()), probs.detach().cpu().numpy()

    def record_reward(self, reward: float) -> None:
        self.rewards.append(float(reward))

    def update_policy(self) -> float:
        if len(self.rewards) == 0 or len(self.log_probs) == 0:
            return 0.0

        returns = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + EPS)

        policy_losses = []
        for log_prob, R in zip(self.log_probs, returns_tensor):
            policy_losses.append(-log_prob * R)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum()
        loss.backward()
        self.optimizer.step()

        loss_value = float(loss.item())

        self.log_probs = []
        self.rewards = []

        return loss_value


# ------------------------------------------------------------
# REVISION GENERATION
# ------------------------------------------------------------

def fallback_revision(user_argument: str, exemplar: str, reason: str = "") -> str:
    reason_clause = ""
    if reason.strip():
        reason_clause = f" Also address this criticism from the user: {reason.strip()}"

    return (
        "Suggested revision:\n\n"
        f"{user_argument.strip()}\n\n"
        "Revised toward stronger persuasion:\n"
        "Narrow the claim, make one mechanism explicit, and replace absolute language with defensible scope. "
        f"Use this retrieved exemplar as guidance: {exemplar.strip()[:450]}..."
        f"{reason_clause}"
    )


def openai_revision(user_argument: str, exemplar: str, reason: str = "") -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    reason_clause = f"\nThe user previously rejected a revision and said: {reason.strip()}" if reason.strip() else ""

    prompt = f"Original argument:\n{user_argument}\n\nExemplar:\n{exemplar}{reason_clause}\n\nRevise the original argument to be more persuasive."

    resp = client.chat.completions.create(
        model="gpt-4o-mini", # FIXED: Use a real model name
        temperature=0.4,
        messages=[
            {"role": "system", "content": "You are a rigorous rhetoric assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def anthropic_revision(user_argument: str, exemplar: str, reason: str = "") -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    reason_clause = f"\nThe user previously rejected a revision and said: {reason.strip()}" if reason.strip() else ""

    prompt = f"Original argument:\n{user_argument}\n\nExemplar:\n{exemplar}{reason_clause}\n\nRevise the original argument."

    resp = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=700,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def generate_revision(user_argument: str, exemplar: str, rejection_reason: str = "") -> str:
    if OPENAI_API_KEY:
        try: return openai_revision(user_argument, exemplar, rejection_reason)
        except Exception as e: return f"{fallback_revision(user_argument, exemplar, rejection_reason)}\n\n[OpenAI fallback: {e}]"
    if ANTHROPIC_API_KEY:
        try: return anthropic_revision(user_argument, exemplar, rejection_reason)
        except Exception as e: return f"{fallback_revision(user_argument, exemplar, rejection_reason)}\n\n[Anthropic fallback: {e}]"
    return fallback_revision(user_argument, exemplar, rejection_reason)


# ------------------------------------------------------------
# SESSION HELPERS
# ------------------------------------------------------------

def init_state():
    if "startup_error" not in st.session_state: st.session_state.startup_error = None
    if "bank" not in st.session_state: st.session_state.bank = load_bank(BANK_PATH)
    if "embedder" not in st.session_state:
        try: st.session_state.embedder = get_embedder()
        except Exception as e:
            st.session_state.embedder = None
            st.session_state.startup_error = str(e)

    if "faiss_index" not in st.session_state:
        if st.session_state.get("embedder") is not None:
            try:
                index, embs = build_faiss_index(st.session_state.bank, st.session_state.embedder)
                st.session_state.faiss_index = index
                st.session_state.bank_embs = embs
            except Exception as e:
                st.session_state.faiss_index = None
                st.session_state.startup_error = str(e)
    
    if "agent" not in st.session_state: st.session_state.agent = RetrievalAgent()
    if "current_argument" not in st.session_state: st.session_state.current_argument = ""
    if "current_topic" not in st.session_state: st.session_state.current_topic = "general"
    if "current_iteration" not in st.session_state: st.session_state.current_iteration = 0
    if "candidate_set" not in st.session_state: st.session_state.candidate_set = []
    if "selected_candidate_local_idx" not in st.session_state: st.session_state.selected_candidate_local_idx = None
    if "selected_candidate_probs" not in st.session_state: st.session_state.selected_candidate_probs = None
    if "latest_revision" not in st.session_state: st.session_state.latest_revision = ""
    if "latest_rejection_reason" not in st.session_state: st.session_state.latest_rejection_reason = ""
    if "accepted_history" not in st.session_state: st.session_state.accepted_history = []
    if "latest_loss" not in st.session_state: st.session_state.latest_loss = 0.0

def rebuild_index():
    index, embs = build_faiss_index(st.session_state.bank, st.session_state.embedder)
    st.session_state.faiss_index = index
    st.session_state.bank_embs = embs
    save_bank(st.session_state.bank, BANK_PATH)

def start_new_session(argument: str, topic: str):
    st.session_state.current_argument = argument.strip()
    st.session_state.current_topic = topic.strip() or "general"
    st.session_state.current_iteration = 0
    st.session_state.candidate_set = []
    st.session_state.selected_candidate_local_idx = None
    st.session_state.selected_candidate_probs = None
    st.session_state.latest_revision = ""
    st.session_state.latest_rejection_reason = ""
    st.session_state.accepted_history = []
    st.session_state.latest_loss = 0.0

def produce_revision():
    argument = st.session_state.current_argument
    if not argument.strip(): return
    st.session_state.current_iteration += 1
    candidates = retrieve_candidates(argument, st.session_state.bank, st.session_state.faiss_index, st.session_state.embedder)
    st.session_state.candidate_set = candidates
    if not candidates:
        st.session_state.latest_revision = fallback_revision(argument, "", st.session_state.latest_rejection_reason)
        return
    idx, probs = st.session_state.agent.choose_candidate(candidates)
    st.session_state.selected_candidate_local_idx = idx
    st.session_state.selected_candidate_probs = probs
    st.session_state.latest_revision = generate_revision(argument, candidates[idx]["entry"].text, st.session_state.latest_rejection_reason)

def apply_feedback(reward: float, reason: str = ""):
    st.session_state.agent.record_reward(reward)
    st.session_state.latest_loss = st.session_state.agent.update_policy()
    idx = st.session_state.selected_candidate_local_idx
    if idx is not None and st.session_state.candidate_set:
        entry = st.session_state.bank[st.session_state.candidate_set[idx]["bank_idx"]]
        entry.reward_sum += reward
        entry.reward_count += 1
        entry.avg_reward = entry.reward_sum / max(entry.reward_count, 1)
    if reason.strip(): st.session_state.latest_rejection_reason = reason.strip()
    rebuild_index()

def finalize_winner(convinced_reason: str):
    final_text = st.session_state.latest_revision.strip()
    if not final_text: return
    new_entry = ArgumentEntry(str(uuid.uuid4()), final_text, st.session_state.current_topic, 2.0, 1, 2.0, st.session_state.current_iteration, convinced_reason.strip())
    st.session_state.bank.append(new_entry)
    st.session_state.accepted_history.append(final_text)
    rebuild_index()

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
init_state()

st.title(APP_TITLE)
st.caption("Iterative argument optimization with semantic retrieval, FAISS, and REINFORCE-style retrieval updates.")
if st.session_state.get("startup_error"):
    st.error("Startup problem: embedding model / FAISS initialization failed.")
    st.code(st.session_state["startup_error"])

with st.sidebar:
    st.header("Settings")
    st.write(f"Bank size: **{len(st.session_state.bank)}**")
    st.write(f"FAISS vectors: **{st.session_state.faiss_index.ntotal if st.session_state.faiss_index else 0}**")
    st.write(f"Last policy loss: **{st.session_state.latest_loss:.4f}**")
    if st.button("Reset bank to defaults"):
        st.session_state.bank = default_bank()
        rebuild_index()
    st.divider()
    st.download_button("Download bank JSON", json.dumps([x.to_dict() for x in st.session_state.bank], indent=2), "bank.json")

left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    topic = st.text_input("Topic", value=st.session_state.current_topic)
    argument = st.text_area("Argument / Claim", value=st.session_state.current_argument, height=250)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start / Reset Session"): start_new_session(argument, topic)
    with c2:
        if st.button("Generate Revision"):
            if argument.strip():
                if argument.strip() != st.session_state.current_argument: start_new_session(argument, topic)
                produce_revision()
    st.write(f"Iterations: **{st.session_state.current_iteration}**")

with right:
    st.subheader("Revision Output")
    if st.session_state.latest_revision:
        st.text_area("Latest Revision", value=st.session_state.latest_revision, height=350)
        reject_reason = st.text_input("Reason if rejected", value=st.session_state.latest_rejection_reason)
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("👍 Reward"): apply_feedback(1.0, "")
        with b2:
            if st.button("👎 Penalize"): apply_feedback(-1.0, reject_reason)
        with b3:
            if st.button("✅ Finalize"):
                apply_feedback(2.0, "")
                finalize_winner(reject_reason or "Convinced.")
    else: st.info("No revision yet.")

st.divider()
st.subheader("Retrieved Candidates")
if st.session_state.candidate_set:
    st.dataframe([{"rank": i+1, "similarity": round(c["similarity"], 4), "text": c["entry"].text[:100]} for i, c in enumerate(st.session_state.candidate_set)])


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embs = embs.astype(np.float32)
    embs = normalize_rows(embs)
    return embs


def build_faiss_index(bank: List[ArgumentEntry], model: SentenceTransformer) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    texts = [entry.text for entry in bank]
    if len(texts) == 0:
        index = faiss.IndexFlatIP(EMBED_DIM)
        return index, np.zeros((0, EMBED_DIM), dtype=np.float32)

    embs = embed_texts(texts, model)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, embs


def retrieve_candidates(
    query_text: str,
    bank: List[ArgumentEntry],
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
    top_k: int = TOP_K_RETRIEVAL,
) -> List[Dict[str, Any]]:
    if len(bank) == 0 or index.ntotal == 0:
        return []

    q = embed_texts([query_text], model)
    sims, idxs = index.search(q, min(top_k, len(bank)))
    sims = sims[0]
    idxs = idxs[0]

    out = []
    for sim, idx in zip(sims, idxs):
        if idx < 0: continue
        entry = bank[int(idx)]
        out.append({
            "bank_idx": int(idx),
            "similarity": float(sim),
            "entry": entry,
        })
    return out


# ------------------------------------------------------------
# POLICY NETWORK (REINFORCE-STYLE)
# ------------------------------------------------------------

class RetrievalPolicy(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = POLICY_HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        scores = self.fc3(x).squeeze(-1)  
        return scores


class RetrievalAgent:
    def __init__(self, learning_rate: float = LEARNING_RATE, gamma: float = GAMMA):
        self.policy = RetrievalPolicy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def candidate_features(self, candidates: List[Dict[str, Any]]) -> np.ndarray:
        feats = []
        for c in candidates:
            entry = c["entry"]
            similarity = c["similarity"]
            avg_reward = entry.avg_reward
            reward_count_norm = min(entry.reward_count / 10.0, 1.0)
            inv_iterations_norm = 1.0 / max(entry.iterations_to_convergence, 1)
            feats.append([
                similarity,
                avg_reward,
                reward_count_norm,
                inv_iterations_norm,
            ])
        return np.array(feats, dtype=np.float32)

    def choose_candidate(self, candidates: List[Dict[str, Any]]) -> Tuple[int, np.ndarray]:
        feats = self.candidate_features(candidates)
        feats_tensor = torch.FloatTensor(feats)

        scores = self.policy(feats_tensor)
        probs = torch.softmax(scores, dim=0)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.log_probs.append(log_prob)

        return int(action.item()), probs.detach().cpu().numpy()

    def record_reward(self, reward: float) -> None:
        self.rewards.append(float(reward))

    def update_policy(self) -> float:
        if len(self.rewards) == 0 or len(self.log_probs) == 0:
            return 0.0

        returns = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + EPS)

        policy_losses = []
        for log_prob, R in zip(self.log_probs, returns_tensor):
            policy_losses.append(-log_prob * R)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum()
        loss.backward()
        self.optimizer.step()

        loss_value = float(loss.item())

        self.log_probs = []
        self.rewards = []

        return loss_value


# ------------------------------------------------------------
# REVISION GENERATION
# ------------------------------------------------------------

def fallback_revision(user_argument: str, exemplar: str, reason: str = "") -> str:
    reason_clause = ""
    if reason.strip():
        reason_clause = f" Also address this criticism from the user: {reason.strip()}"

    return (
        "Suggested revision:\n\n"
        f"{user_argument.strip()}\n\n"
        "Revised toward stronger persuasion:\n"
        "Narrow the claim, make one mechanism explicit, and replace absolute language with defensible scope. "
        f"Use this retrieved exemplar as guidance: {exemplar.strip()[:450]}..."
        f"{reason_clause}"
    )


def openai_revision(user_argument: str, exemplar: str, reason: str = "") -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    reason_clause = f"\nThe user previously rejected a revision and said: {reason.strip()}" if reason.strip() else ""

    prompt = f"Original argument:\n{user_argument}\n\nExemplar:\n{exemplar}{reason_clause}\n\nRevise the original argument to be more persuasive."

    resp = client.chat.completions.create(
        model="gpt-4o-mini", # FIXED: Use a real model name
        temperature=0.4,
        messages=[
            {"role": "system", "content": "You are a rigorous rhetoric assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def anthropic_revision(user_argument: str, exemplar: str, reason: str = "") -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    reason_clause = f"\nThe user previously rejected a revision and said: {reason.strip()}" if reason.strip() else ""

    prompt = f"Original argument:\n{user_argument}\n\nExemplar:\n{exemplar}{reason_clause}\n\nRevise the original argument."

    resp = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=700,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def generate_revision(user_argument: str, exemplar: str, rejection_reason: str = "") -> str:
    if OPENAI_API_KEY:
        try: return openai_revision(user_argument, exemplar, rejection_reason)
        except Exception as e: return f"{fallback_revision(user_argument, exemplar, rejection_reason)}\n\n[OpenAI fallback: {e}]"
    if ANTHROPIC_API_KEY:
        try: return anthropic_revision(user_argument, exemplar, rejection_reason)
        except Exception as e: return f"{fallback_revision(user_argument, exemplar, rejection_reason)}\n\n[Anthropic fallback: {e}]"
    return fallback_revision(user_argument, exemplar, rejection_reason)


# ------------------------------------------------------------
# SESSION HELPERS
# ------------------------------------------------------------

def init_state():
    # Initialize ALL keys immediately so the UI doesn't crash on first-run
    defaults = {
        "startup_error": None,
        "bank": [],
        "embedder": None,
        "faiss_index": None,
        "bank_embs": None,
        "agent": RetrievalAgent(),
        "current_argument": "",
        "current_topic": "general",
        "current_iteration": 0,
        "candidate_set": [],
        "selected_candidate_local_idx": None,
        "selected_candidate_probs": None,
        "latest_revision": "",
        "latest_rejection_reason": "",
        "accepted_history": [],
        "latest_loss": 0.0
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Now attempt the "heavy" loading
    if not st.session_state.bank:
        st.session_state.bank = load_bank(BANK_PATH)

    if st.session_state.embedder is None:
        try:
            st.session_state.embedder = get_embedder()
            idx, embs = build_faiss_index(st.session_state.bank, st.session_state.embedder)
            st.session_state.faiss_index = idx
            st.session_state.bank_embs = embs
        except Exception as e:
            st.session_state.startup_error = str(e)
    if "startup_error" not in st.session_state: st.session_state.startup_error = None
    if "bank" not in st.session_state: st.session_state.bank = load_bank(BANK_PATH)
    if "embedder" not in st.session_state:
        try: st.session_state.embedder = get_embedder()
        except Exception as e:
            st.session_state.embedder = None
            st.session_state.startup_error = str(e)

    if "faiss_index" not in st.session_state:
        if st.session_state.get("embedder") is not None:
            try:
                index, embs = build_faiss_index(st.session_state.bank, st.session_state.embedder)
                st.session_state.faiss_index = index
                st.session_state.bank_embs = embs
            except Exception as e:
                st.session_state.faiss_index = None
                st.session_state.startup_error = str(e)
    
    if "agent" not in st.session_state: st.session_state.agent = RetrievalAgent()
    if "current_argument" not in st.session_state: st.session_state.current_argument = ""
    if "current_topic" not in st.session_state: st.session_state.current_topic = "general"
    if "current_iteration" not in st.session_state: st.session_state.current_iteration = 0
    if "candidate_set" not in st.session_state: st.session_state.candidate_set = []
    if "selected_candidate_local_idx" not in st.session_state: st.session_state.selected_candidate_local_idx = None
    if "selected_candidate_probs" not in st.session_state: st.session_state.selected_candidate_probs = None
    if "latest_revision" not in st.session_state: st.session_state.latest_revision = ""
    if "latest_rejection_reason" not in st.session_state: st.session_state.latest_rejection_reason = ""
    if "accepted_history" not in st.session_state: st.session_state.accepted_history = []
    if "latest_loss" not in st.session_state: st.session_state.latest_loss = 0.0

def rebuild_index():
    index, embs = build_faiss_index(st.session_state.bank, st.session_state.embedder)
    st.session_state.faiss_index = index
    st.session_state.bank_embs = embs
    save_bank(st.session_state.bank, BANK_PATH)

def start_new_session(argument: str, topic: str):
    st.session_state.current_argument = argument.strip()
    st.session_state.current_topic = topic.strip() or "general"
    st.session_state.current_iteration = 0
    st.session_state.candidate_set = []
    st.session_state.selected_candidate_local_idx = None
    st.session_state.selected_candidate_probs = None
    st.session_state.latest_revision = ""
    st.session_state.latest_rejection_reason = ""
    st.session_state.accepted_history = []
    st.session_state.latest_loss = 0.0

def produce_revision():
    argument = st.session_state.current_argument
    if not argument.strip(): return
    st.session_state.current_iteration += 1
    candidates = retrieve_candidates(argument, st.session_state.bank, st.session_state.faiss_index, st.session_state.embedder)
    st.session_state.candidate_set = candidates
    if not candidates:
        st.session_state.latest_revision = fallback_revision(argument, "", st.session_state.latest_rejection_reason)
        return
    idx, probs = st.session_state.agent.choose_candidate(candidates)
    st.session_state.selected_candidate_local_idx = idx
    st.session_state.selected_candidate_probs = probs
    st.session_state.latest_revision = generate_revision(argument, candidates[idx]["entry"].text, st.session_state.latest_rejection_reason)

def apply_feedback(reward: float, reason: str = ""):
    st.session_state.agent.record_reward(reward)
    st.session_state.latest_loss = st.session_state.agent.update_policy()
    idx = st.session_state.selected_candidate_local_idx
    if idx is not None and st.session_state.candidate_set:
        entry = st.session_state.bank[st.session_state.candidate_set[idx]["bank_idx"]]
        entry.reward_sum += reward
        entry.reward_count += 1
        entry.avg_reward = entry.reward_sum / max(entry.reward_count, 1)
    if reason.strip(): st.session_state.latest_rejection_reason = reason.strip()
    rebuild_index()

def finalize_winner(convinced_reason: str):
    final_text = st.session_state.latest_revision.strip()
    if not final_text: return
    new_entry = ArgumentEntry(str(uuid.uuid4()), final_text, st.session_state.current_topic, 2.0, 1, 2.0, st.session_state.current_iteration, convinced_reason.strip())
    st.session_state.bank.append(new_entry)
    st.session_state.accepted_history.append(final_text)
    rebuild_index()

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
init_state()

st.title(APP_TITLE)
st.caption("Iterative argument optimization with semantic retrieval, FAISS, and REINFORCE-style retrieval updates.")
if st.session_state.get("startup_error"):
    st.error("Startup problem: embedding model / FAISS initialization failed.")
    st.code(st.session_state["startup_error"])

with st.sidebar:
    st.header("Settings")
    st.write(f"Bank size: **{len(st.session_state.bank)}**")
    
    # Safe access: if faiss_index is None, show 0 instead of crashing
    ntotal = st.session_state.faiss_index.ntotal if st.session_state.faiss_index else 0
    st.write(f"FAISS vectors: **{ntotal}**")
    
    st.write(f"Last policy loss: **{st.session_state.latest_loss:.4f}**")

left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    topic = st.text_input("Topic", value=st.session_state.current_topic)
    argument = st.text_area("Argument / Claim", value=st.session_state.current_argument, height=250)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start / Reset Session"): start_new_session(argument, topic)
    with c2:
        if st.button("Generate Revision"):
            if argument.strip():
                if argument.strip() != st.session_state.current_argument: start_new_session(argument, topic)
                produce_revision()
    st.write(f"Iterations: **{st.session_state.current_iteration}**")

with right:
    st.subheader("Revision Output")
    if st.session_state.latest_revision:
        st.text_area("Latest Revision", value=st.session_state.latest_revision, height=350)
        reject_reason = st.text_input("Reason if rejected", value=st.session_state.latest_rejection_reason)
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("👍 Reward"): apply_feedback(1.0, "")
        with b2:
            if st.button("👎 Penalize"): apply_feedback(-1.0, reject_reason)
        with b3:
            if st.button("✅ Finalize"):
                apply_feedback(2.0, "")
                finalize_winner(reject_reason or "Convinced.")
    else: st.info("No revision yet.")

st.divider()
st.subheader("Retrieved Candidates")
if st.session_state.candidate_set:
    st.dataframe([{"rank": i+1, "similarity": round(c["similarity"], 4), "text": c["entry"].text[:100]} for i, c in enumerate(st.session_state.candidate_set)])