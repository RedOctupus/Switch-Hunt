"""程序化音效管理器。"""
from __future__ import annotations

import math

import numpy as np
import pygame

from switch_hunt.constants import TILE_SIZE
from switch_hunt.enums import GhostState


class SoundManager:
    """程序化音效与背景配乐管理器（无需外部音频文件）"""

    SR = 22050  # 采样率

    def __init__(self):
        self.enabled = True
        self.initialized = False
        self._sounds = {}
        self._bg_sound = None
        self._bg_channel = None
        self._tension_timer = 0.0
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.SR, size=-16, channels=2, buffer=1024)
            self._generate_all()
            self._bg_channel = pygame.mixer.Channel(7)
            self.initialized = True
        except Exception as e:
            print(f"[Sound] 初始化失败: {e}")

    # ── 基础工具 ────────────────────────────────────────────────
    def _make(self, wave):
        """float32 [-1,1] → pygame.Sound (stereo int16)"""
        arr = np.clip(wave, -1.0, 1.0)
        arr16 = (arr * 26000).astype(np.int16)
        stereo = np.ascontiguousarray(np.column_stack([arr16, arr16]))
        return pygame.sndarray.make_sound(stereo)

    def _t(self, dur):
        return np.linspace(0, dur, int(self.SR * dur), False)

    # ── 音效生成 ─────────────────────────────────────────────────
    def _generate_all(self):
        sr = self.SR

        # 1. 光源激活：高频电流裂变感
        t = self._t(0.35)
        env = np.exp(-t * 12)
        w = 0.55 * np.sin(2 * np.pi * 880 * t) * env
        w += 0.25 * np.sin(2 * np.pi * 1760 * t) * env * np.exp(-t * 18)
        w += np.random.uniform(-0.12, 0.12, len(t)) * env
        self._sounds['light'] = self._make(w)

        # 2. 定身：冰冻扫频下降
        t = self._t(0.5)
        freq_sweep = 600 - 400 * t / 0.5
        phase = np.cumsum(2 * np.pi * freq_sweep / sr)
        env = np.exp(-t * 4)
        w = 0.5 * np.sin(phase) * env
        w += 0.15 * np.sin(2 * np.pi * 3000 * t) * np.exp(-t * 25)
        self._sounds['stun'] = self._make(w)

        # 3. 收集宝藏：上行音阶铃声
        t = self._t(0.65)
        freqs = [523, 659, 784, 1047]   # C5 E5 G5 C6
        w = np.zeros(len(t))
        for i, f in enumerate(freqs):
            s = int(i * 0.09 * sr)
            env_t = np.linspace(0, 0.65 - i * 0.09, len(t) - s, False)
            w[s:] += 0.32 * np.sin(2 * np.pi * f * env_t) * np.exp(-env_t * 5)
        self._sounds['treasure'] = self._make(w)

        # 4. 游戏失败：深沉轰鸣
        t = self._t(1.6)
        env = np.exp(-t * 1.8)
        w = 0.5 * np.sin(2 * np.pi * 55 * t) * env
        w += 0.3 * np.sin(2 * np.pi * 80 * t) * env
        w += 0.08 * np.random.uniform(-1, 1, len(t)) * np.exp(-t * 6)
        self._sounds['game_over'] = self._make(w)

        # 5. 胜利：上行琶音
        t = self._t(1.4)
        vfreqs = [392, 523, 659, 784, 1047]  # G4 C5 E5 G5 C6
        w = np.zeros(len(t))
        for i, f in enumerate(vfreqs):
            s = int(i * 0.17 * sr)
            if s >= len(t):
                break
            e = min(len(t), s + int(0.45 * sr))
            et = np.linspace(0, (e - s) / sr, e - s, False)
            w[s:e] += 0.38 * np.sin(2 * np.pi * f * et) * np.exp(-et * 4)
        self._sounds['victory'] = self._make(w)

        # 6. 紧张心跳：两声低频重击
        t = self._t(0.42)
        e1 = np.exp(-t * 28)
        e2 = np.zeros(len(t))
        mid = int(0.19 * sr)
        if mid < len(t):
            e2[mid:] = np.exp(-np.linspace(0, 0.23, len(t) - mid) * 28)
        w = 0.65 * np.sin(2 * np.pi * 58 * t) * e1
        w += 0.55 * np.sin(2 * np.pi * 58 * t) * e2
        self._sounds['heartbeat'] = self._make(w)

        # 7. 背景配乐：暗黑无缝循环（~6秒）
        t = self._t(6.0)
        w = np.zeros(len(t))
        for freq, vol in [(55, 0.28), (82.5, 0.18), (110, 0.11), (73.4, 0.08)]:
            wobble = 0.6 * np.sin(2 * np.pi * 0.22 * t)
            w += vol * np.sin(2 * np.pi * freq * t + wobble)
        # 缓慢震颤 LFO
        w *= 0.58 + 0.42 * np.sin(2 * np.pi * 0.14 * t)
        # 低频脉冲纹理
        pulse = np.maximum(0, np.sin(2 * np.pi * 0.75 * t)) ** 4
        w += 0.10 * np.sin(2 * np.pi * 55 * t) * pulse
        w /= (np.max(np.abs(w)) + 1e-8)
        w *= 0.38
        self._bg_sound = self._make(w)

    # ── 公共接口 ──────────────────────────────────────────────────
    def play(self, name, volume=1.0):
        if not self.enabled or not self.initialized:
            return
        s = self._sounds.get(name)
        if s:
            s.set_volume(volume)
            s.play()

    def start_music(self):
        if self.initialized and self._bg_sound and self._bg_channel and self.enabled:
            self._bg_channel.set_volume(0.45)
            self._bg_channel.play(self._bg_sound, loops=-1)

    def stop_music(self):
        if self.initialized and self._bg_channel:
            self._bg_channel.stop()

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self.start_music()
        else:
            self.stop_music()

    def update(self, dt, game):
        """根据鬼的距离动态触发心跳紧张音效"""
        if not self.enabled or not self.initialized:
            return
        self._tension_timer -= dt
        if self._tension_timer > 0 or not game.ghosts:
            return
        player = game.player
        closest = float('inf')
        for ghost in game.ghosts:
            if ghost.state != GhostState.STUNNED:
                d = math.sqrt((player.pos[0] - ghost.pos[0])**2 +
                              (player.pos[1] - ghost.pos[1])**2)
                closest = min(closest, d)
        danger_range = 5 * TILE_SIZE
        if closest <= danger_range:
            self.play('heartbeat', volume=0.6 + 0.4 * (1 - closest / danger_range))
            # 越近心跳越快
            self._tension_timer = 0.35 + (closest / danger_range) * 0.75
        else:
            self._tension_timer = 0.5

