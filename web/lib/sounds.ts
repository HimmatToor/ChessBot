// Move sounds synthesised with the Web Audio API: no audio files to host or
// license. A filtered noise burst gives the wooden "thock"; sine/triangle
// tones add body and the check/game-end chimes.

export type SoundName = "move" | "capture" | "check" | "end";

const STORAGE_KEY = "chessbot:muted";

let ctx: AudioContext | null = null;
let noise: AudioBuffer | null = null;
let muted = readMuted();
const listeners = new Set<() => void>();

function readMuted(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false; // storage blocked (private mode etc.)
  }
}

export function isMuted(): boolean {
  return muted;
}

export function setMuted(value: boolean) {
  muted = value;
  try {
    localStorage.setItem(STORAGE_KEY, value ? "1" : "0");
  } catch {
    // not persisted, but still applies for this session
  }
  listeners.forEach((listener) => listener());
}

export function subscribeMuted(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function playSound(name: SoundName) {
  if (muted || typeof window === "undefined") return;
  // Browsers only allow audio after a user gesture; moves are gestures,
  // so the context is created lazily on the first one.
  ctx ??= new AudioContext();
  if (ctx.state === "suspended") void ctx.resume();
  const t = ctx.currentTime;

  switch (name) {
    case "move":
      click(ctx, t, 1800, 0.5, 0.06);
      tone(ctx, t, 220, 0.25, 0.07);
      break;
    case "capture":
      click(ctx, t, 1100, 0.8, 0.1);
      tone(ctx, t, 150, 0.4, 0.12);
      click(ctx, t + 0.035, 2400, 0.3, 0.05);
      break;
    case "check":
      click(ctx, t, 1800, 0.5, 0.06);
      tone(ctx, t + 0.02, 988, 0.15, 0.18);
      tone(ctx, t + 0.12, 1319, 0.15, 0.22);
      break;
    case "end":
      click(ctx, t, 1800, 0.5, 0.06);
      [523.25, 659.25, 783.99].forEach((freq, i) =>
        tone(ctx!, t + 0.08 + i * 0.12, freq, 0.2, 0.6, "triangle"),
      );
      break;
  }
}

function envelope(ac: AudioContext, t: number, peak: number, decay: number): GainNode {
  const gain = ac.createGain();
  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.exponentialRampToValueAtTime(peak, t + 0.004);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + decay);
  return gain;
}

function click(ac: AudioContext, t: number, freq: number, peak: number, decay: number) {
  if (!noise) {
    noise = ac.createBuffer(1, Math.floor(ac.sampleRate * 0.2), ac.sampleRate);
    const data = noise.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  }
  const source = ac.createBufferSource();
  source.buffer = noise;
  const filter = ac.createBiquadFilter();
  filter.type = "bandpass";
  filter.frequency.value = freq;
  filter.Q.value = 1.2;
  source.connect(filter).connect(envelope(ac, t, peak, decay)).connect(ac.destination);
  source.start(t);
  source.stop(t + decay);
}

function tone(
  ac: AudioContext,
  t: number,
  freq: number,
  peak: number,
  decay: number,
  type: OscillatorType = "sine",
) {
  const osc = ac.createOscillator();
  osc.type = type;
  osc.frequency.value = freq;
  osc.connect(envelope(ac, t, peak, decay)).connect(ac.destination);
  osc.start(t);
  osc.stop(t + decay);
}
