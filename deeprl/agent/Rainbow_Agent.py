



from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
from collections import deque

class RainbowActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def _set_up(self):
        self.config.atoms = tensor(self.config.atoms)

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock, torch.no_grad():
            probs, _ = self._network(config.state_normalizer(self._state))
        q_values = (probs * self.config.atoms).sum(-1)
        q_values = to_np(q_values).flatten()

        if self.config.noisy:
            action = np.argmax(q_values)
        else:
            if self._total_steps < config.exploration_steps \
                    or np.random.rand() < config.random_action_prob():
                action = np.random.randint(0, len(q_values))
            else:
                action = np.argmax(q_values)
        
        next_state, reward, done, info = self._task.step([action])
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        return entry


class RainbowAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()
        config.atoms = np.linspace(config.categorical_v_min,
                                   config.categorical_v_max, config.categorical_n_atoms)

        self.replay = config.replay_fn()
        self.actor = RainbowActor(config)

        self.network = config.network_fn()
        self.network.train()
        self.network.share_memory()

        self.target_network = config.network_fn()
        self.target_network.train()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)
        self.atoms = tensor(config.atoms)
        self.delta_atom = (config.categorical_v_max - config.categorical_v_min) / float(config.categorical_n_atoms - 1)

        self.tracker = deque(maxlen=config.nstep)

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        if np.random.rand() > self.config.test_epsilon:
            state = self.config.state_normalizer(state)
            prob, _ = self.network(state)
            q = (prob * self.atoms).sum(-1)
            action = np.argmax(to_np(q).flatten())
        else:
            action = self.config.eval_env.action_space.sample()

        self.config.state_normalizer.unset_read_only()
        return [action]

    def step(self):
        config = self.config

        if config.noisy:
            self.network.reset_noise()
            self.target_network.reset_noise()

        transitions = self.actor.step()

        experiences = []
        for state, action, reward, next_state, done, info in transitions:
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            self.tracker.append([state, action, reward, done])

            R = 0
            for *_, r, d in reversed(self.tracker):
                R += r + self.config.discount * (1 - d) * R
            experiences.append(self.tracker[0][:2] + [R, next_state, done])

            if done:
                self.record_online_return(info)
                self.tracker.clear()

        self.replay.feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps:
            beta = config.beta_schedule()
            experiences = self.replay.sample(beta=beta)
            states, actions, rewards, next_states, terminals, *extras = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            actions = actions.long()


            with torch.no_grad():
                
                prob_next, _ = self.target_network(next_states)

                if config.double:
                    prob_next_online, _ = self.network(next_states)
                    actions_next = prob_next_online.mul(self.atoms).sum(dim=-1).argmax(dim=-1)
                else:
                    actions_next = prob_next.mul(self.atoms).sum(dim=-1).argmax(dim=-1)

                prob_next = prob_next.detach()
                q_next = (prob_next * self.atoms).sum(-1)
                prob_next = prob_next[self.batch_indices, actions_next, :]

                rewards = tensor(rewards).unsqueeze(-1)
                terminals = tensor(terminals).unsqueeze(-1)
                atoms_next = rewards + self.config.discount * (1 - terminals) * self.atoms.view(1, -1)

                atoms_next.clamp_(self.config.categorical_v_min, self.config.categorical_v_max)
                b = (atoms_next - self.config.categorical_v_min) / self.delta_atom
                l = b.floor()
                u = b.ceil()
                d_m_l = (u + (l == u).float() - b) * prob_next
                d_m_u = (b - l) * prob_next
                target_prob = tensor(np.zeros(prob_next.size()))
                for i in range(target_prob.size(0)):
                    target_prob[i].index_add_(0, l[i].long(), d_m_l[i])
                    target_prob[i].index_add_(0, u[i].long(), d_m_u[i])

                
            
            _, log_prob = self.network(states)
            log_prob = log_prob[self.batch_indices, actions, :]
            error = 0 - target_prob.mul(log_prob).sum(dim=-1)

            if extras is not None:
                idxes, weights = extras
                idxes = to_np(idxes.int()).flatten().tolist()
                weights = to_np(error).flatten().tolist()
                self.replay.update_priorities(idxes, weights)

            loss = error.mean()

            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

            self.logger.store(Loss=loss.item())

        if self.total_steps  % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())