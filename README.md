# SteelMan

An iterative argument revision tool using human-in-the-loop reinforcement learning. You paste an argument, a policy network selects a rhetorical transform, an LLM applies it, and you rate the result. Ratings train the policy in real time and persist across sessions.

## How it works

Each revision cycle runs two retrieval streams in parallel:

- **Exemplar store:** a FAISS index over finalized argument trajectories. If a past argument with a similar endpoint exists (cosine ≥ 0.75), its initial/final text and best-performing transform are injected into the revision prompt as a before/after example.
- **Transform bank:** a set of rhetorical operations (e.g. *Preempt the Strongest Objection*, *Neutralize Loaded Language*). Each transform accumulates reward observations over time. Once a transform has ≥ 20 observations, a per-transform logistic regression replaces cosine similarity for retrieval, learning where in embedding space the transform tends to succeed, independent of topic. Users can also add their own transforms.

A REINFORCE policy network scores the top-k retrieved candidates and samples one. After you rate the revision (+1 accept, −1 reject, +2 finalise), the policy receives a gradient update, bank stats are updated, and the argument embedding is appended to the selected transform's observation list.

Rejecting an output and providing a reason triggers an LLM-based transform mutation where a new variant is generated, and enters the bank at generation+1 with zero reward. Users can also add their own transforms. 

Finalising writes the full revision trajectory to the argument store, which feeds future exemplar retrieval.

## Setup

```bash
conda env create -f steelman.yml
conda activate steelman
```

Create a `.env` and add at least one API key:

```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## Run

```bash
streamlit run steelman.py
```

## Files

| File | Contents |
|------|----------|
| `steelman_bank.json` | Transform bank: all entries including reward stats and observations |
| `steelman_args.json` | Finalized argument store: full trajectories used for exemplar retrieval |
| `steelman_policy.pt` | REINFORCE policy weights and optimizer state |

Delete any of these to reset that component. Deleting `steelman_bank.json` regenerates the default bank of 8 transforms.

## Optional context

The **Add context** toggle exposes fields for a parent argument (what you are responding to), audience, venue, and constraints. Context is injected into both the retrieval query and the revision prompt. The parent argument is not checked by the ethical filter as arguing against a harmful position is a legitimate use of this tool.

## Ethical safeguards

Every argument is checked against 7 hard-block categories before any revision is generated. The classifier uses a threshold of 0.8 and is calibrated with explicit few-shot examples distinguishing advocacy from reporting, criticism, and analysis. There is no override path.

## AI disclosure

Claude (Anthropic) was used throughout development for code implementation and debugging, architecture discussion, and writing assistance. All intellectual decisions, problem framing, and system design are the author's own.