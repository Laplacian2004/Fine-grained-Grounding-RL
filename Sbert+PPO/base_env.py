import numpy as np
import gymnasium as gym
from sentence_transformers import SentenceTransformer

MODEL = "models/all-MiniLM-L6-v2"
EMB_DIM = 384


class BaseEnv(gym.Env):
    def __init__(self, seed=None, max_steps=2000):
        super().__init__()
        self.random = np.random.RandomState(seed)

        # Gym spaces
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(EMB_DIM*2 + 4,),  # span_emb + ground_emb + scalars
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)

        # State
        self.t = 0
        self.max_steps = max_steps
        self.span = [0, 1]
        self.ground_text = None
        self.terminate = False

        # Data
        self.sources = []
        self.ground_embs = []
        self.starts = []
        self.ends = []
        self.num_training_data = 0
        self.data_id = 0
        self.source = None
        self.source_len = 0
        self.load_precomp_data("dataset_precomp.npz")
        self.encoder = SentenceTransformer(MODEL)

        # Reward
        self.alpha = 0.1
        self.prev_manhattan_dist = 0
        self.prev_iou = 0

    # --------------------
    # Gym API
    # --------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        idx = self.random.choice(self.num_training_data)
        self.data_id = idx
        self.source = self.sources[idx]
        self.source_len = len(self.source)
        self.span = [0, self.source_len-1]

        self.ground_text = {
            "grounded_text": self.source[self.starts[idx]:self.ends[idx]+1],
            "emb": self.ground_embs[idx],
            "start": int(self.starts[idx]),
            "end": int(self.ends[idx])
        }

        self.t = 0
        self.terminate = False
        self.prev_manhattan_dist = abs(
            self.starts[idx]) + abs(self.ends[idx] - self.source_len + 1)
        self.prev_iou = 0
        obs = {
            "ground_text": self.ground_text["grounded_text"],
            "ground_emb": self.ground_text["emb"],
            "span": tuple(self.span),
            "source": self.source
        }
        return obs, {}

    def step(self, action):
        self.calculate_span(action)
        self.t += 1
        truncated = False
        if self.t >= self.max_steps:
            truncated = True
            self.terminate = True

        reward = self.calculate_reward()

        obs = {
            "ground_text": self.ground_text["grounded_text"],
            "ground_emb": self.ground_text["emb"],
            "span": tuple(self.span),
            "source": self.source
        }

        info = {"env_info": self._build_info()}
        return obs, reward, self.terminate, truncated, info

    # --------------------
    # Observation encoding
    # --------------------
    def encode_obs(self, obs):
        ground_emb = obs["ground_emb"]
        source = obs["source"]
        start, end = obs["span"]

        span_text = source[start:end + 1] if 0 <= start <= end < len(source) else ""
        span_emb = self.encoder.encode(
                span_text, convert_to_numpy=True, normalize_embeddings=True
            )
        # Scalars
        N = len(source)
        span_len = end - start + 1
        cosine_sim = float(np.dot(span_emb, ground_emb))
        #  print(cosine_sim)

        scalars = np.array([
            start / N,
            end / N,
            span_len / N,
            cosine_sim
        ], dtype=np.float32)

        return np.concatenate([span_emb, ground_emb, scalars]).astype(np.float32)

    # --------------------
    # Span update
    # --------------------
    def calculate_span(self, action):
        if action == 0:
            self.span[0] = min(self.span[1], self.span[0]+1)
        elif action == 1:
            self.span[1] = max(self.span[0], self.span[1]-1)
        elif action == 2:
            self.terminate = True

    # --------------------
    # Reward calculation
    # --------------------
    def calculate_reward(self):
        ps, pe = self.span[0], self.span[1]
        gs, ge = self.ground_text["start"], self.ground_text["end"]
        iou = self._compute_iou(ps, pe, gs, ge)
        if iou == 0:
            self.terminate = True
        reward = 0
        if self.terminate:
            reward = 3 if iou >= 0.6 else -100
        else:
            iou_diff = iou - self.prev_iou
            reward = 1 if iou_diff > 0 else -1
            self.prev_iou = iou
        '''
        else:
            new_manhattan_dist = abs(gs - ps) + abs(ge - pe)
            reward += self.alpha * \
                (self.prev_manhattan_dist - new_manhattan_dist) / self.source_len
            self.prev_manhattan_dist = new_manhattan_dist
        '''
        return reward

    # --------------------
    # Utility / Info
    # --------------------
    def _compute_iou(self, ps, pe, gs, ge):
        inter = max(0, min(pe, ge) - max(ps, gs) + 1)
        union = (pe - ps + 1) + (ge - gs + 1) - inter
        return inter / union if union > 0 else 0.0

    def _build_info(self):
        start, end = self.span
        gs, ge = self.ground_text["start"], self.ground_text["end"]

        info = {
            "span_start": start,
            "span_end": end,
            "ground_start": gs,
            "ground_end": ge,
            "iou": self._compute_iou(start, end, gs, ge)
        }
        return info

    def seed(self, seed):
        self.random = np.random.RandomState(seed)

    def render(self):
        pass

    def load_precomp_data(self, file):
        data = np.load(file, allow_pickle=True)
        self.ground_embs = data["ground_embs"]
        self.sources = data["sources"]
        self.starts = data["starts"]
        self.ends = data["ends"]
        self.num_training_data = len(self.sources)
        print("Loaded", self.num_training_data, "precomputed spans")
